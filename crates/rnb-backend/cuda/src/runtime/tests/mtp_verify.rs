use super::*;

#[test]
fn qwen35_mtp_verify_buffer_plan_counts_window_buffers() {
    let plan = qwen35_mtp_verify_buffer_plan(3, 4096, 1).unwrap();

    assert_eq!(plan.window_tokens, 3);
    assert_eq!(plan.hidden_dim, 4096);
    assert_eq!(plan.prefix_count, 1);
    assert_eq!(plan.token_id_bytes, 3 * std::mem::size_of::<u32>());
    assert_eq!(plan.target_token_bytes, 3 * std::mem::size_of::<u32>());
    assert_eq!(plan.hidden_row_bytes, 3 * 4096 * std::mem::size_of::<f32>());
    assert_eq!(plan.scratch_hidden_bytes, plan.hidden_row_bytes);
    assert!(plan.total_device_bytes() > plan.hidden_row_bytes);
}

#[test]
fn qwen35_mtp_verify_prefix_indices_must_be_inside_window() {
    assert!(mtp_verify::validate_mtp_verify_prefix_tokens(3, &[1, 2, 3]).is_ok());
    assert!(mtp_verify::validate_mtp_verify_prefix_tokens(3, &[]).is_ok());

    let zero = mtp_verify::validate_mtp_verify_prefix_tokens(3, &[0]).unwrap_err();
    assert!(zero.contains("must be > 0"));

    let out_of_window = mtp_verify::validate_mtp_verify_prefix_tokens(3, &[4]).unwrap_err();
    assert!(out_of_window.contains("must be <= window_tokens"));
}

#[test]
fn cuda_qwen35_mtp_verify_buffers_allocate_from_plan() {
    let _guard = runtime_test_lock();
    let Ok(mut state) = CudaState::open() else {
        eprintln!("skipping CUDA MTP verify buffer allocation test: CUDA driver unavailable");
        return;
    };
    let plan = qwen35_mtp_verify_buffer_plan(2, 16, 1).unwrap();

    let buffers = match state.ensure_mtp_verify_buffers(&plan) {
        Ok(buffers) => buffers,
        Err(err) => {
            eprintln!("skipping CUDA MTP verify buffer allocation test: {err}");
            return;
        }
    };

    assert_ne!(buffers.token_ids_dev, 0);
    assert_ne!(buffers.target_tokens_dev, 0);
    assert_ne!(buffers.hidden_rows_dev, 0);
    assert_ne!(buffers.scratch_hidden_dev, 0);
    assert_ne!(buffers.prefix_indices_dev, 0);
    assert_eq!(buffers.plan.window_tokens, 2);
    assert_eq!(buffers.plan.hidden_dim, 16);
}

#[test]
fn cuda_qwen35_mtp_verify_stage_uploads_tokens_and_prefix_indices() {
    let _guard = runtime_test_lock();
    let Ok(mut state) = CudaState::open() else {
        eprintln!("skipping CUDA MTP verify staging test: CUDA driver unavailable");
        return;
    };
    let verify_tokens = [10_u32, 11, 12];
    let prefix_tokens = [1_usize, 2];
    let plan = qwen35_mtp_verify_buffer_plan(verify_tokens.len(), 8, prefix_tokens.len()).unwrap();

    let buffers = match state.stage_mtp_verify_window(&plan, &verify_tokens, &prefix_tokens) {
        Ok(buffers) => buffers,
        Err(err) => {
            eprintln!("skipping CUDA MTP verify staging test: {err}");
            return;
        }
    };

    let mut uploaded_tokens = vec![0_u32; verify_tokens.len()];
    let mut uploaded_prefix = vec![0_u32; prefix_tokens.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                uploaded_tokens.as_mut_ptr().cast::<libc::c_void>(),
                buffers.token_ids_dev,
                std::mem::size_of_val(uploaded_tokens.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                uploaded_prefix.as_mut_ptr().cast::<libc::c_void>(),
                buffers.prefix_indices_dev,
                std::mem::size_of_val(uploaded_prefix.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    assert_eq!(uploaded_tokens, verify_tokens);
    assert_eq!(uploaded_prefix, [1_u32, 2]);
}

#[test]
fn cuda_qwen35_mtp_verify_stage_allows_empty_prefix_indices() {
    let _guard = runtime_test_lock();
    let Ok(mut state) = CudaState::open() else {
        eprintln!("skipping CUDA MTP verify empty-prefix staging test: CUDA driver unavailable");
        return;
    };
    let verify_tokens = [10_u32];
    let plan = qwen35_mtp_verify_buffer_plan(verify_tokens.len(), 8, 0).unwrap();

    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("empty prefix staging should not allocate zero-byte CUDA buffers");

    assert_ne!(buffers.token_ids_dev, 0);
    assert_eq!(buffers.prefix_indices_dev, 0);
}

#[test]
fn cuda_qwen35_mtp_verify_collects_target_tokens_and_hidden_rows() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let plan = qwen35_mtp_verify_buffer_plan(2, 4, 0).unwrap();
    let buffers = state
        .ensure_mtp_verify_buffers(&plan)
        .expect("allocate verify buffers");
    let target_tokens = [101_u32, 202];
    let hidden_rows = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.target_tokens_dev,
                target_tokens.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(target_tokens.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let result = state
        .collect_mtp_verify_result(&plan)
        .expect("collect verify result");

    assert_eq!(result.target_tokens, target_tokens);
    assert_eq!(result.mtp_hidden_rows, hidden_rows);
    assert_eq!(result.hidden_dim, 4);
}

#[test]
fn cuda_qwen35_mtp_verify_embedding_stage_writes_hidden_rows() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let blocks_per_row = hidden_dim / 256;
    let token_embd_rows = 4usize;
    let token_embd = make_test_q4k_weights(1, token_embd_rows, blocks_per_row, 233)
        .pop()
        .unwrap();
    let verify_tokens = [2_u32, 0, 3];
    let plan = qwen35_mtp_verify_buffer_plan(verify_tokens.len(), hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");

    state
        .stage_mtp_verify_token_embeddings_q4k(
            &buffers,
            &token_embd,
            token_embd_rows,
            hidden_dim,
            &verify_tokens,
        )
        .expect("stage embedding rows");

    let mut actual = vec![0.0f32; verify_tokens.len() * hidden_dim];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected = Vec::with_capacity(actual.len());
    for &token_id in &verify_tokens {
        let row_idx = token_id as usize;
        let row = &token_embd[row_idx * blocks_per_row * 144..(row_idx + 1) * blocks_per_row * 144];
        expected.extend(cpu_q4k_dequant_row(row, blocks_per_row));
    }

    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-6,
            "MTP verify embedding stage mismatch at {idx}: expected {expected}, actual {actual}, diff {diff}"
        );
    }
}

#[test]
fn cuda_qwen35_mtp_verify_embedding_stage_accepts_q6k_rows() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let blocks_per_row = hidden_dim / 256;
    let token_embd_rows = 4usize;
    let token_embd = make_test_q6k_weights(1, token_embd_rows, blocks_per_row, 337)
        .pop()
        .unwrap();
    let verify_tokens = [1_u32, 3, 0];
    let plan = qwen35_mtp_verify_buffer_plan(verify_tokens.len(), hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");

    state
        .stage_mtp_verify_token_embeddings_q6k(
            &buffers,
            &token_embd,
            token_embd_rows,
            hidden_dim,
            &verify_tokens,
        )
        .expect("stage Q6_K embedding rows");

    let mut actual = vec![0.0f32; verify_tokens.len() * hidden_dim];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected = Vec::with_capacity(actual.len());
    for &token_id in &verify_tokens {
        let row_idx = token_id as usize;
        let row = &token_embd[row_idx * blocks_per_row * 210..(row_idx + 1) * blocks_per_row * 210];
        expected.extend(cpu_q6k_dequant_row(row, blocks_per_row));
    }

    assert_close_rows(
        "MTP verify Q6_K embedding stage",
        &actual,
        &expected,
        1.0e-6,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_hidden_rows_rms_norms_window_to_scratch() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 3usize;
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.013671875)
        .collect::<Vec<_>>();
    let norm_weight = (0..hidden_dim)
        .map(|i| 0.5 + ((i % 13) as f32) * 0.02734375)
        .collect::<Vec<_>>();
    let verify_tokens = [7_u32, 8, 9];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    state
        .stage_mtp_verify_hidden_rows_rms_norm(&buffers, &norm_weight, 1.0e-5, false)
        .expect("rms norm verify rows");

    let mut actual = vec![0.0f32; window_tokens * hidden_dim];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.scratch_hidden_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected = Vec::with_capacity(actual.len());
    for hidden in hidden_rows.chunks_exact(hidden_dim) {
        expected.extend(cpu_rms_norm(hidden, &norm_weight, 1.0e-5, false));
    }

    assert_close_rows("MTP verify hidden RMSNorm rows", &actual, &expected, 1.0e-5);
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_input_projections_match_cpu_reference() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 3usize;
    let qkv_rows = 32usize;
    let gate_rows = 24usize;
    let alpha_rows = 5usize;
    let beta_rows = 5usize;
    let blocks_per_row = hidden_dim / 256;
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.01171875)
        .collect::<Vec<_>>();
    let attn_norm = (0..hidden_dim)
        .map(|i| 0.625 + ((i % 17) as f32) * 0.01953125)
        .collect::<Vec<_>>();
    let qkv = make_test_q4k_weights(1, qkv_rows, blocks_per_row, 317)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, gate_rows, blocks_per_row, 331)
        .pop()
        .unwrap();
    let alpha = make_test_q4k_weights(1, alpha_rows, blocks_per_row, 347)
        .pop()
        .unwrap();
    let beta = make_test_q4k_weights(1, beta_rows, blocks_per_row, 359)
        .pop()
        .unwrap();
    let verify_tokens = [11_u32, 12, 13];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let projection_buffers = state
        .stage_mtp_verify_gdn_input_projections_q4k(
            &buffers,
            mtp_verify::Qwen35MtpGdnProjectionRequest {
                attn_norm: &attn_norm,
                qkv_q4k: &qkv,
                qkv_quant: 12,
                qkv_rows,
                qkv_cols: hidden_dim,
                gate_q4k: &gate,
                gate_rows,
                gate_cols: hidden_dim,
                alpha_q4k: &alpha,
                alpha_f32: &[],
                alpha_quant: GGML_Q4_K,
                alpha_rows,
                alpha_cols: hidden_dim,
                beta_q4k: &beta,
                beta_f32: &[],
                beta_quant: GGML_Q4_K,
                beta_rows,
                beta_cols: hidden_dim,
                norm_eps: 1.0e-5,
            },
        )
        .expect("stage GDN input projections");

    let mut actual_qkv = vec![0.0f32; window_tokens * qkv_rows];
    let mut actual_gate = vec![0.0f32; window_tokens * gate_rows];
    let mut actual_alpha = vec![0.0f32; window_tokens * alpha_rows];
    let mut actual_beta = vec![0.0f32; window_tokens * beta_rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_qkv.as_mut_ptr().cast::<libc::c_void>(),
                projection_buffers.qkv_dev,
                std::mem::size_of_val(actual_qkv.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_gate.as_mut_ptr().cast::<libc::c_void>(),
                projection_buffers.gate_dev,
                std::mem::size_of_val(actual_gate.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_alpha.as_mut_ptr().cast::<libc::c_void>(),
                projection_buffers.alpha_dev,
                std::mem::size_of_val(actual_alpha.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_beta.as_mut_ptr().cast::<libc::c_void>(),
                projection_buffers.beta_dev,
                std::mem::size_of_val(actual_beta.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected_qkv = Vec::with_capacity(actual_qkv.len());
    let mut expected_gate = Vec::with_capacity(actual_gate.len());
    let mut expected_alpha = Vec::with_capacity(actual_alpha.len());
    let mut expected_beta = Vec::with_capacity(actual_beta.len());
    for hidden in hidden_rows.chunks_exact(hidden_dim) {
        let normed = cpu_rms_norm(hidden, &attn_norm, 1.0e-5, false);
        expected_qkv.extend(cpu_q4k_gemv_rows(&qkv, qkv_rows, blocks_per_row, &normed));
        expected_gate.extend(cpu_q4k_gemv_rows(&gate, gate_rows, blocks_per_row, &normed));
        expected_alpha.extend(cpu_q4k_gemv_rows(
            &alpha,
            alpha_rows,
            blocks_per_row,
            &normed,
        ));
        expected_beta.extend(cpu_q4k_gemv_rows(&beta, beta_rows, blocks_per_row, &normed));
    }

    assert_close_rows("MTP verify GDN qkv", &actual_qkv, &expected_qkv, 0.05);
    assert_close_rows("MTP verify GDN gate", &actual_gate, &expected_gate, 0.05);
    assert_close_rows("MTP verify GDN alpha", &actual_alpha, &expected_alpha, 0.05);
    assert_close_rows("MTP verify GDN beta", &actual_beta, &expected_beta, 0.05);
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_input_projections_accept_f32_alpha_beta() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 2usize;
    let qkv_rows = 16usize;
    let gate_rows = 12usize;
    let alpha_rows = 4usize;
    let beta_rows = 4usize;
    let blocks_per_row = hidden_dim / 256;
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.009765625)
        .collect::<Vec<_>>();
    let attn_norm = (0..hidden_dim)
        .map(|i| 0.75 + ((i % 11) as f32) * 0.013671875)
        .collect::<Vec<_>>();
    let qkv = make_test_q4k_weights(1, qkv_rows, blocks_per_row, 367)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, gate_rows, blocks_per_row, 379)
        .pop()
        .unwrap();
    let alpha = (0..alpha_rows * hidden_dim)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.00390625)
        .collect::<Vec<_>>();
    let beta = (0..beta_rows * hidden_dim)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.0029296875)
        .collect::<Vec<_>>();
    let verify_tokens = [17_u32, 18];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let projection_buffers = state
        .stage_mtp_verify_gdn_input_projections_q4k(
            &buffers,
            mtp_verify::Qwen35MtpGdnProjectionRequest {
                attn_norm: &attn_norm,
                qkv_q4k: &qkv,
                qkv_quant: GGML_Q4_K,
                qkv_rows,
                qkv_cols: hidden_dim,
                gate_q4k: &gate,
                gate_rows,
                gate_cols: hidden_dim,
                alpha_q4k: &[],
                alpha_f32: &alpha,
                alpha_quant: GGML_F32,
                alpha_rows,
                alpha_cols: hidden_dim,
                beta_q4k: &[],
                beta_f32: &beta,
                beta_quant: GGML_F32,
                beta_rows,
                beta_cols: hidden_dim,
                norm_eps: 1.0e-5,
            },
        )
        .expect("stage GDN input projections with F32 alpha/beta");

    let mut actual_alpha = vec![0.0f32; window_tokens * alpha_rows];
    let mut actual_beta = vec![0.0f32; window_tokens * beta_rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_alpha.as_mut_ptr().cast::<libc::c_void>(),
                projection_buffers.alpha_dev,
                std::mem::size_of_val(actual_alpha.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_beta.as_mut_ptr().cast::<libc::c_void>(),
                projection_buffers.beta_dev,
                std::mem::size_of_val(actual_beta.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected_alpha = Vec::with_capacity(actual_alpha.len());
    let mut expected_beta = Vec::with_capacity(actual_beta.len());
    for hidden in hidden_rows.chunks_exact(hidden_dim) {
        let normed = cpu_rms_norm(hidden, &attn_norm, 1.0e-5, false);
        expected_alpha.extend(cpu_f32_gemv_rows(&alpha, alpha_rows, hidden_dim, &normed));
        expected_beta.extend(cpu_f32_gemv_rows(&beta, beta_rows, hidden_dim, &normed));
    }

    assert_close_rows(
        "MTP verify GDN alpha F32",
        &actual_alpha,
        &expected_alpha,
        1.0e-4,
    );
    assert_close_rows(
        "MTP verify GDN beta F32",
        &actual_beta,
        &expected_beta,
        1.0e-4,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_attention_qkv_projections_match_cpu_reference() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 3usize;
    let q_rows = 48usize;
    let k_rows = 24usize;
    let v_rows = 24usize;
    let blocks_per_row = hidden_dim / 256;
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.0107421875)
        .collect::<Vec<_>>();
    let attn_norm = (0..hidden_dim)
        .map(|i| 0.75 + ((i % 19) as f32) * 0.015625)
        .collect::<Vec<_>>();
    let q = make_test_q4k_weights(1, q_rows, blocks_per_row, 401)
        .pop()
        .unwrap();
    let k = make_test_q4k_weights(1, k_rows, blocks_per_row, 409)
        .pop()
        .unwrap();
    let v = make_test_q4k_weights(1, v_rows, blocks_per_row, 419)
        .pop()
        .unwrap();
    let verify_tokens = [21_u32, 22, 23];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let projection_buffers = state
        .stage_mtp_verify_attention_qkv_projections_q4k(
            &buffers,
            mtp_verify::Qwen35MtpAttentionQkvProjectionRequest {
                attn_norm: &attn_norm,
                q_q4k: &q,
                q_quant: 12,
                q_rows,
                q_cols: hidden_dim,
                k_q4k: &k,
                k_quant: 12,
                k_rows,
                k_cols: hidden_dim,
                v_q4k: &v,
                v_quant: 12,
                v_rows,
                v_cols: hidden_dim,
                norm_eps: 1.0e-5,
            },
        )
        .expect("stage attention qkv projections");

    let mut actual_q = vec![0.0f32; window_tokens * q_rows];
    let mut actual_k = vec![0.0f32; window_tokens * k_rows];
    let mut actual_v = vec![0.0f32; window_tokens * v_rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_q.as_mut_ptr().cast::<libc::c_void>(),
                projection_buffers.q_dev,
                std::mem::size_of_val(actual_q.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_k.as_mut_ptr().cast::<libc::c_void>(),
                projection_buffers.k_dev,
                std::mem::size_of_val(actual_k.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_v.as_mut_ptr().cast::<libc::c_void>(),
                projection_buffers.v_dev,
                std::mem::size_of_val(actual_v.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected_q = Vec::with_capacity(actual_q.len());
    let mut expected_k = Vec::with_capacity(actual_k.len());
    let mut expected_v = Vec::with_capacity(actual_v.len());
    for hidden in hidden_rows.chunks_exact(hidden_dim) {
        let normed = cpu_rms_norm(hidden, &attn_norm, 1.0e-5, false);
        expected_q.extend(cpu_q4k_gemv_rows(&q, q_rows, blocks_per_row, &normed));
        expected_k.extend(cpu_q4k_gemv_rows(&k, k_rows, blocks_per_row, &normed));
        expected_v.extend(cpu_q4k_gemv_rows(&v, v_rows, blocks_per_row, &normed));
    }

    assert_close_rows("MTP verify attention q", &actual_q, &expected_q, 0.05);
    assert_close_rows("MTP verify attention k", &actual_k, &expected_k, 0.05);
    assert_close_rows("MTP verify attention v", &actual_v, &expected_v, 0.05);
}

#[test]
fn cuda_qwen35_mtp_verify_attention_qk_norm_rope_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 3usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let q_rows = num_heads * head_dim;
    let kv_rows = num_kv_heads * head_dim;
    let blocks_per_row = hidden_dim / 256;
    let pos_start = 11usize;
    let rope_theta = 10000.0f32;
    let rope_dim = 64usize;
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.009765625)
        .collect::<Vec<_>>();
    let attn_norm = (0..hidden_dim)
        .map(|i| 0.7 + ((i % 17) as f32) * 0.01171875)
        .collect::<Vec<_>>();
    let q_norm = (0..head_dim)
        .map(|i| 0.8 + ((i % 13) as f32) * 0.0078125)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.9 + ((i % 11) as f32) * 0.0068359375)
        .collect::<Vec<_>>();
    let q = make_test_q4k_weights(1, q_rows, blocks_per_row, 431)
        .pop()
        .unwrap();
    let k = make_test_q4k_weights(1, kv_rows, blocks_per_row, 439)
        .pop()
        .unwrap();
    let v = make_test_q4k_weights(1, kv_rows, blocks_per_row, 443)
        .pop()
        .unwrap();
    let verify_tokens = [31_u32, 32, 33];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let projection_buffers = state
        .stage_mtp_verify_attention_qkv_projections_q4k(
            &buffers,
            mtp_verify::Qwen35MtpAttentionQkvProjectionRequest {
                attn_norm: &attn_norm,
                q_q4k: &q,
                q_quant: 12,
                q_rows,
                q_cols: hidden_dim,
                k_q4k: &k,
                k_quant: 12,
                k_rows: kv_rows,
                k_cols: hidden_dim,
                v_q4k: &v,
                v_quant: 12,
                v_rows: kv_rows,
                v_cols: hidden_dim,
                norm_eps: 1.0e-5,
            },
        )
        .expect("stage attention qkv projections");

    let post_buffers = state
        .stage_mtp_verify_attention_qk_norm_rope(
            &projection_buffers,
            mtp_verify::Qwen35MtpAttentionQkNormRopeRequest {
                q_norm: &q_norm,
                k_norm: &k_norm,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_dim,
                rope_neox: false,
                rope_theta,
                pos_start,
                norm_eps: 1.0e-5,
                q_unit_offset: false,
                k_unit_offset: false,
                v_no_scale_norm: false,
            },
        )
        .expect("stage attention qk norm rope");

    let mut actual_q = vec![0.0f32; window_tokens * q_rows];
    let mut actual_k_bits = vec![0u16; window_tokens * kv_rows];
    let mut actual_v_bits = vec![0u16; window_tokens * kv_rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_q.as_mut_ptr().cast::<libc::c_void>(),
                post_buffers.q_dev,
                std::mem::size_of_val(actual_q.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_k_bits.as_mut_ptr().cast::<libc::c_void>(),
                post_buffers.k_bits_dev,
                std::mem::size_of_val(actual_k_bits.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_v_bits.as_mut_ptr().cast::<libc::c_void>(),
                post_buffers.v_bits_dev,
                std::mem::size_of_val(actual_v_bits.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();
    let (collected_k_bits, collected_v_bits) = state
        .collect_mtp_verify_attention_window_kv_bits(&post_buffers)
        .expect("collect current attention window K/V bits");
    assert_eq!(collected_k_bits, actual_k_bits);
    assert_eq!(collected_v_bits, actual_v_bits);
    let (deferred_k_bits, deferred_v_bits) = state
        .collect_mtp_verify_attention_window_kv_bits_deferred(&post_buffers)
        .expect("queue deferred current attention window K/V bits");
    state.stream_synchronize().unwrap();
    assert_eq!(deferred_k_bits, actual_k_bits);
    assert_eq!(deferred_v_bits, actual_v_bits);

    let mut q_raw = Vec::with_capacity(window_tokens * q_rows);
    let mut k_raw = Vec::with_capacity(window_tokens * kv_rows);
    let mut v_raw = Vec::with_capacity(window_tokens * kv_rows);
    for hidden in hidden_rows.chunks_exact(hidden_dim) {
        let normed = cpu_rms_norm(hidden, &attn_norm, 1.0e-5, false);
        q_raw.extend(cpu_q4k_gemv_rows(&q, q_rows, blocks_per_row, &normed));
        k_raw.extend(cpu_q4k_gemv_rows(&k, kv_rows, blocks_per_row, &normed));
        v_raw.extend(cpu_q4k_gemv_rows(&v, kv_rows, blocks_per_row, &normed));
    }
    let expected_q = cpu_qk_norm_rope_select(
        &q_raw,
        &q_norm,
        window_tokens,
        num_heads,
        head_dim,
        rope_dim,
        false,
        pos_start,
        1.0e-5,
        rope_theta,
        false,
    );
    let expected_k = cpu_qk_norm_rope_select(
        &k_raw,
        &k_norm,
        window_tokens,
        num_kv_heads,
        head_dim,
        rope_dim,
        false,
        pos_start,
        1.0e-5,
        rope_theta,
        false,
    );
    let actual_k = actual_k_bits
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect::<Vec<_>>();
    let actual_v = actual_v_bits
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect::<Vec<_>>();
    assert_close_rows_abs_rel(
        "MTP verify attention q postprocess",
        &actual_q,
        &expected_q,
        0.25,
        0.01,
    );
    assert_close_rows_abs_rel(
        "MTP verify attention k postprocess",
        &actual_k,
        &expected_k,
        0.05,
        0.01,
    );
    assert_close_rows_abs_rel(
        "MTP verify attention v postprocess",
        &actual_v,
        &v_raw,
        0.05,
        0.01,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_attention_output_hd256_window_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 4usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let q_rows = num_heads * head_dim;
    let kv_rows = num_kv_heads * head_dim;
    let blocks_per_row = hidden_dim / 256;
    let pos_start = 13usize;
    let rope_theta = 10000.0f32;
    let window = 3usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.0087890625)
        .collect::<Vec<_>>();
    let attn_norm = (0..hidden_dim)
        .map(|i| 0.65 + ((i % 17) as f32) * 0.0107421875)
        .collect::<Vec<_>>();
    let q_norm = (0..head_dim)
        .map(|i| 0.75 + ((i % 13) as f32) * 0.0068359375)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.85 + ((i % 11) as f32) * 0.005859375)
        .collect::<Vec<_>>();
    let q = make_test_q4k_weights(1, q_rows, blocks_per_row, 457)
        .pop()
        .unwrap();
    let k = make_test_q4k_weights(1, kv_rows, blocks_per_row, 461)
        .pop()
        .unwrap();
    let v = make_test_q4k_weights(1, kv_rows, blocks_per_row, 463)
        .pop()
        .unwrap();
    let verify_tokens = [41_u32, 42, 43, 44];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let projection_buffers = state
        .stage_mtp_verify_attention_qkv_projections_q4k(
            &buffers,
            mtp_verify::Qwen35MtpAttentionQkvProjectionRequest {
                attn_norm: &attn_norm,
                q_q4k: &q,
                q_quant: 12,
                q_rows,
                q_cols: hidden_dim,
                k_q4k: &k,
                k_quant: 12,
                k_rows: kv_rows,
                k_cols: hidden_dim,
                v_q4k: &v,
                v_quant: 12,
                v_rows: kv_rows,
                v_cols: hidden_dim,
                norm_eps: 1.0e-5,
            },
        )
        .expect("stage attention qkv projections");
    let post_buffers = state
        .stage_mtp_verify_attention_qk_norm_rope(
            &projection_buffers,
            mtp_verify::Qwen35MtpAttentionQkNormRopeRequest {
                q_norm: &q_norm,
                k_norm: &k_norm,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_dim: head_dim,
                rope_neox: true,
                rope_theta,
                pos_start,
                norm_eps: 1.0e-5,
                q_unit_offset: false,
                k_unit_offset: false,
                v_no_scale_norm: false,
            },
        )
        .expect("stage attention qk norm rope");

    let attention_buffers = state
        .stage_mtp_verify_attention_output_hd256_window(
            &post_buffers,
            mtp_verify::Qwen35MtpAttentionOutputRequest {
                num_heads,
                num_kv_heads,
                scale,
                window,
            },
        )
        .expect("stage attention output");

    let mut actual = vec![0.0f32; window_tokens * q_rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                attention_buffers.attn_out_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut q_raw = Vec::with_capacity(window_tokens * q_rows);
    let mut k_raw = Vec::with_capacity(window_tokens * kv_rows);
    let mut v_raw = Vec::with_capacity(window_tokens * kv_rows);
    for hidden in hidden_rows.chunks_exact(hidden_dim) {
        let normed = cpu_rms_norm(hidden, &attn_norm, 1.0e-5, false);
        q_raw.extend(cpu_q4k_gemv_rows(&q, q_rows, blocks_per_row, &normed));
        k_raw.extend(cpu_q4k_gemv_rows(&k, kv_rows, blocks_per_row, &normed));
        v_raw.extend(cpu_q4k_gemv_rows(&v, kv_rows, blocks_per_row, &normed));
    }
    let q_post = cpu_qk_norm_rope_neox(
        &q_raw,
        &q_norm,
        window_tokens,
        num_heads,
        head_dim,
        pos_start,
        1.0e-5,
        rope_theta,
        None,
        false,
    );
    let k_post = cpu_qk_norm_rope_neox(
        &k_raw,
        &k_norm,
        window_tokens,
        num_kv_heads,
        head_dim,
        pos_start,
        1.0e-5,
        rope_theta,
        None,
        false,
    );
    let k_f16 = k_post
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let v_f16 = v_raw
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for t in 0..window_tokens {
        let global_pos = t;
        let start = (global_pos + 1).saturating_sub(window);
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q_post[q_off..q_off + head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1 - start);
            for j in start..=global_pos {
                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let dot = q_row
                    .iter()
                    .zip(k_f16[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| a * half::f16::from_bits(*b).to_f32())
                    .sum::<f32>()
                    * scale;
                scores.push((j, dot));
            }
            let max_score = scores
                .iter()
                .map(|(_, score)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|(_, score)| (*score - max_score).exp())
                .sum::<f32>();
            for (j, score) in scores {
                let p = (score - max_score).exp() / denom;
                let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let out_off = t * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    expected[out_off + d] += p * half::f16::from_bits(v_f16[v_off + d]).to_f32();
                }
            }
        }
    }

    assert_close_rows_abs_rel(
        "MTP verify attention output hd256 window",
        &actual,
        &expected,
        0.05,
        0.01,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_attention_output_hd256_includes_prior_device_kv() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 3usize;
    let prior_tokens = 2usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let q_rows = num_heads * head_dim;
    let kv_rows = num_kv_heads * head_dim;
    let blocks_per_row = hidden_dim / 256;
    let pos_start = 17usize;
    let rope_theta = 10000.0f32;
    let window = prior_tokens + window_tokens;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.0078125)
        .collect::<Vec<_>>();
    let prior_k_bits = (0..prior_tokens * kv_rows)
        .map(|i| half::f16::from_f32(((i as f32 % 31.0) - 15.0) * 0.01171875).to_bits())
        .collect::<Vec<_>>();
    let prior_v_bits = (0..prior_tokens * kv_rows)
        .map(|i| half::f16::from_f32(((i as f32 % 29.0) - 14.0) * 0.013671875).to_bits())
        .collect::<Vec<_>>();
    let prior_k_dev = state
        .mem_alloc(std::mem::size_of_val(prior_k_bits.as_slice()))
        .expect("allocate prior k bits");
    let prior_v_dev = state
        .mem_alloc(std::mem::size_of_val(prior_v_bits.as_slice()))
        .expect("allocate prior v bits");
    let attn_norm = (0..hidden_dim)
        .map(|i| 0.72 + ((i % 13) as f32) * 0.0068359375)
        .collect::<Vec<_>>();
    let q_norm = (0..head_dim)
        .map(|i| 0.82 + ((i % 11) as f32) * 0.005859375)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.76 + ((i % 7) as f32) * 0.0048828125)
        .collect::<Vec<_>>();
    let q = make_test_q4k_weights(1, q_rows, blocks_per_row, 527)
        .pop()
        .unwrap();
    let k = make_test_q4k_weights(1, kv_rows, blocks_per_row, 541)
        .pop()
        .unwrap();
    let v = make_test_q4k_weights(1, kv_rows, blocks_per_row, 547)
        .pop()
        .unwrap();
    let verify_tokens = [71_u32, 72, 73];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                prior_k_dev,
                prior_k_bits.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(prior_k_bits.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                prior_v_dev,
                prior_v_bits.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(prior_v_bits.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let projection_buffers = state
        .stage_mtp_verify_attention_qkv_projections_q4k(
            &buffers,
            mtp_verify::Qwen35MtpAttentionQkvProjectionRequest {
                attn_norm: &attn_norm,
                q_q4k: &q,
                q_quant: 12,
                q_rows,
                q_cols: hidden_dim,
                k_q4k: &k,
                k_quant: 12,
                k_rows: kv_rows,
                k_cols: hidden_dim,
                v_q4k: &v,
                v_quant: 12,
                v_rows: kv_rows,
                v_cols: hidden_dim,
                norm_eps: 1.0e-5,
            },
        )
        .expect("stage attention qkv projections");
    let post_buffers = state
        .stage_mtp_verify_attention_qk_norm_rope(
            &projection_buffers,
            mtp_verify::Qwen35MtpAttentionQkNormRopeRequest {
                q_norm: &q_norm,
                k_norm: &k_norm,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_dim: head_dim,
                rope_neox: true,
                rope_theta,
                pos_start,
                norm_eps: 1.0e-5,
                q_unit_offset: false,
                k_unit_offset: false,
                v_no_scale_norm: false,
            },
        )
        .expect("stage attention qk norm rope");

    let attention_buffers = state
        .stage_mtp_verify_attention_output_hd256_prior_window(
            &post_buffers,
            mtp_verify::Qwen35MtpAttentionOutputWithPriorRequest {
                prior_k_bits_dev: prior_k_dev,
                prior_v_bits_dev: prior_v_dev,
                prior_tokens,
                num_heads,
                num_kv_heads,
                scale,
                window,
            },
        )
        .expect("stage attention output with prior");

    let mut actual = vec![0.0f32; window_tokens * q_rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                attention_buffers.attn_out_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();
    unsafe {
        state.api.mem_free(prior_k_dev).unwrap();
        state.api.mem_free(prior_v_dev).unwrap();
    }

    let mut q_raw = Vec::with_capacity(window_tokens * q_rows);
    let mut k_raw = Vec::with_capacity(window_tokens * kv_rows);
    let mut v_raw = Vec::with_capacity(window_tokens * kv_rows);
    for hidden in hidden_rows.chunks_exact(hidden_dim) {
        let normed = cpu_rms_norm(hidden, &attn_norm, 1.0e-5, false);
        q_raw.extend(cpu_q4k_gemv_rows(&q, q_rows, blocks_per_row, &normed));
        k_raw.extend(cpu_q4k_gemv_rows(&k, kv_rows, blocks_per_row, &normed));
        v_raw.extend(cpu_q4k_gemv_rows(&v, kv_rows, blocks_per_row, &normed));
    }
    let q_post = cpu_qk_norm_rope_neox(
        &q_raw,
        &q_norm,
        window_tokens,
        num_heads,
        head_dim,
        pos_start,
        1.0e-5,
        rope_theta,
        None,
        false,
    );
    let k_post = cpu_qk_norm_rope_neox(
        &k_raw,
        &k_norm,
        window_tokens,
        num_kv_heads,
        head_dim,
        pos_start,
        1.0e-5,
        rope_theta,
        None,
        false,
    );
    let current_k_bits = k_post
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let current_v_bits = v_raw
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let mut all_k = prior_k_bits
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect::<Vec<_>>();
    all_k.extend(
        current_k_bits
            .iter()
            .map(|&x| half::f16::from_bits(x).to_f32()),
    );
    let mut all_v = prior_v_bits
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect::<Vec<_>>();
    all_v.extend(
        current_v_bits
            .iter()
            .map(|&x| half::f16::from_bits(x).to_f32()),
    );
    let heads_per_group = num_heads / num_kv_heads;
    let mut expected = vec![0.0f32; actual.len()];
    for t in 0..window_tokens {
        let global_pos = prior_tokens + t;
        let start = (global_pos + 1).saturating_sub(window);
        for h in 0..num_heads {
            let kv_h = h / heads_per_group;
            let q_off = t * num_heads * head_dim + h * head_dim;
            let q_row = &q_post[q_off..q_off + head_dim];
            let mut scores = Vec::with_capacity(global_pos + 1 - start);
            for j in start..=global_pos {
                let k_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let dot = q_row
                    .iter()
                    .zip(all_k[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
                scores.push((j, dot));
            }
            let max_score = scores
                .iter()
                .map(|(_, score)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let denom = scores
                .iter()
                .map(|(_, score)| (*score - max_score).exp())
                .sum::<f32>();
            for (j, score) in scores {
                let p = (score - max_score).exp() / denom;
                let v_off = j * num_kv_heads * head_dim + kv_h * head_dim;
                let out_off = t * num_heads * head_dim + h * head_dim;
                for d in 0..head_dim {
                    expected[out_off + d] += p * all_v[v_off + d];
                }
            }
        }
    }

    assert_close_rows_abs_rel(
        "MTP verify attention output hd256 prior window",
        &actual,
        &expected,
        0.05,
        0.01,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_attention_output_hd256_prior_matches_decode_kernel() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let window_tokens = 3usize;
    let prior_tokens = 5usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let q_rows = num_heads * head_dim;
    let kv_rows = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q = (0..window_tokens * q_rows)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.00634765625)
        .collect::<Vec<_>>();
    let prior_k_bits = (0..prior_tokens * kv_rows)
        .map(|i| half::f16::from_f32(((i as f32 % 37.0) - 18.0) * 0.0107421875).to_bits())
        .collect::<Vec<_>>();
    let prior_v_bits = (0..prior_tokens * kv_rows)
        .map(|i| half::f16::from_f32(((i as f32 % 41.0) - 20.0) * 0.0087890625).to_bits())
        .collect::<Vec<_>>();
    let current_k_bits = (0..window_tokens * kv_rows)
        .map(|i| half::f16::from_f32(((i as f32 % 31.0) - 15.0) * 0.0126953125).to_bits())
        .collect::<Vec<_>>();
    let current_v_bits = (0..window_tokens * kv_rows)
        .map(|i| half::f16::from_f32(((i as f32 % 29.0) - 14.0) * 0.01171875).to_bits())
        .collect::<Vec<_>>();

    let q_dev = state
        .mem_alloc(std::mem::size_of_val(q.as_slice()))
        .expect("allocate q");
    let k_dev = state
        .mem_alloc(std::mem::size_of_val(current_k_bits.as_slice()))
        .expect("allocate current k");
    let v_dev = state
        .mem_alloc(std::mem::size_of_val(current_v_bits.as_slice()))
        .expect("allocate current v");
    let prior_k_dev = state
        .mem_alloc(std::mem::size_of_val(prior_k_bits.as_slice()))
        .expect("allocate prior k");
    let prior_v_dev = state
        .mem_alloc(std::mem::size_of_val(prior_v_bits.as_slice()))
        .expect("allocate prior v");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                q_dev,
                q.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(q.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                k_dev,
                current_k_bits.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(current_k_bits.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                v_dev,
                current_v_bits.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(current_v_bits.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                prior_k_dev,
                prior_k_bits.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(prior_k_bits.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                prior_v_dev,
                prior_v_bits.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(prior_v_bits.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let post_buffers = mtp_verify::MtpVerifyAttentionQkNormRopeBuffers {
        q_dev,
        gate_dev: None,
        k_bits_dev: k_dev,
        v_bits_dev: v_dev,
        window_tokens,
        q_rows,
        kv_rows,
        head_dim,
    };
    let attention_buffers = state
        .stage_mtp_verify_attention_output_hd256_prior_window(
            &post_buffers,
            mtp_verify::Qwen35MtpAttentionOutputWithPriorRequest {
                prior_k_bits_dev: prior_k_dev,
                prior_v_bits_dev: prior_v_dev,
                prior_tokens,
                num_heads,
                num_kv_heads,
                scale,
                window: prior_tokens + window_tokens,
            },
        )
        .expect("stage attention output with prior");
    let mut actual = vec![0.0f32; window_tokens * q_rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                attention_buffers.attn_out_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut all_k = prior_k_bits.clone();
    all_k.extend(current_k_bits.iter().copied());
    let mut all_v = prior_v_bits.clone();
    all_v.extend(current_v_bits.iter().copied());
    let mut expected = Vec::with_capacity(actual.len());
    for t in 0..window_tokens {
        let q_start = t * q_rows;
        let kv_len = prior_tokens + t + 1;
        let out = state
            .attention_decode_hd256(
                &q[q_start..q_start + q_rows],
                &all_k[..kv_len * kv_rows],
                &all_v[..kv_len * kv_rows],
                kv_len,
                num_heads,
                num_kv_heads,
                scale,
            )
            .expect("decode attention");
        expected.extend(out);
    }

    unsafe {
        state.api.mem_free(q_dev).unwrap();
        state.api.mem_free(k_dev).unwrap();
        state.api.mem_free(v_dev).unwrap();
        state.api.mem_free(prior_k_dev).unwrap();
        state.api.mem_free(prior_v_dev).unwrap();
    }

    assert_close_rows_abs_rel(
        "MTP verify attention output hd256 prior decode",
        &actual,
        &expected,
        1.0e-6,
        1.0e-6,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_attention_prior_kv_cache_reuses_layer_device_buffers() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let layer_index = 4usize;
    let kv_rows = 8usize;
    let first_tokens = 2usize;
    let extended_tokens = 3usize;
    let first_k = (0..first_tokens * kv_rows)
        .map(|i| half::f16::from_f32((i as f32 + 1.0) * 0.01).to_bits())
        .collect::<Vec<_>>();
    let first_v = (0..first_tokens * kv_rows)
        .map(|i| half::f16::from_f32((i as f32 + 5.0) * 0.01).to_bits())
        .collect::<Vec<_>>();
    let (first_k_dev, first_v_dev) = state
        .stage_mtp_verify_attention_prior_kv_bits_for_layer(
            layer_index,
            &first_k,
            &first_v,
            first_tokens,
            kv_rows,
        )
        .expect("stage first prior kv")
        .expect("prior kv device buffers");

    let mut extended_k = first_k.clone();
    extended_k.extend(
        (first_tokens * kv_rows..extended_tokens * kv_rows)
            .map(|i| half::f16::from_f32((i as f32 + 1.0) * 0.01).to_bits()),
    );
    let mut extended_v = first_v.clone();
    extended_v.extend(
        (first_tokens * kv_rows..extended_tokens * kv_rows)
            .map(|i| half::f16::from_f32((i as f32 + 5.0) * 0.01).to_bits()),
    );
    let (extended_k_dev, extended_v_dev) = state
        .stage_mtp_verify_attention_prior_kv_bits_for_layer(
            layer_index,
            &extended_k,
            &extended_v,
            extended_tokens,
            kv_rows,
        )
        .expect("stage extended prior kv")
        .expect("extended prior kv device buffers");

    assert_eq!(extended_k_dev, first_k_dev);
    assert_eq!(extended_v_dev, first_v_dev);
    let cache = state
        .mtp_verify_attention_prior_kv
        .iter()
        .find(|cache| cache.layer_index == layer_index)
        .expect("layer prior kv cache");
    assert_eq!(cache.cached_tokens, extended_tokens);
    assert_eq!(cache.host_k_bits, extended_k);
    assert_eq!(cache.host_v_bits, extended_v);

    let mut actual_k = vec![0u16; extended_k.len()];
    let mut actual_v = vec![0u16; extended_v.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_k.as_mut_ptr().cast::<libc::c_void>(),
                extended_k_dev,
                std::mem::size_of_val(actual_k.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_v.as_mut_ptr().cast::<libc::c_void>(),
                extended_v_dev,
                std::mem::size_of_val(actual_v.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    assert_eq!(actual_k, extended_k);
    assert_eq!(actual_v, extended_v);
}

#[test]
fn cuda_qwen35_mtp_verify_attention_o_projection_residual_ffn_norm_stays_device_resident() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let attn_dim = hidden_dim * 2;
    let window_tokens = 3usize;
    let blocks_per_row = attn_dim / 256;
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 53.0) - 26.0) * 0.0068359375)
        .collect::<Vec<_>>();
    let attention_out = (0..window_tokens * attn_dim)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.009765625)
        .collect::<Vec<_>>();
    let o_proj = make_test_q4k_weights(1, hidden_dim, blocks_per_row, 467)
        .pop()
        .unwrap();
    let ffn_norm = (0..hidden_dim)
        .map(|i| 0.7 + ((i % 19) as f32) * 0.0087890625)
        .collect::<Vec<_>>();
    let verify_tokens = [51_u32, 52, 53];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    let attention_out_dev = state
        .mem_alloc(std::mem::size_of_val(attention_out.as_slice()))
        .expect("allocate attention output");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                attention_out_dev,
                attention_out.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(attention_out.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let attention_buffers = mtp_verify::MtpVerifyAttentionOutputBuffers {
        k_f32_dev: 0,
        v_f32_dev: 0,
        attn_out_dev: attention_out_dev,
        window_tokens,
        q_rows: attn_dim,
        kv_rows: 0,
        head_dim: 256,
    };

    state
        .stage_mtp_verify_attention_o_projection_residual_ffn_norm_q4k(
            &buffers,
            &attention_buffers,
            &o_proj,
            GGML_Q4_K,
            hidden_dim,
            attn_dim,
            &ffn_norm,
            1.0e-5,
        )
        .expect("stage attention o projection residual ffn norm");

    let mut actual_hidden = vec![0.0f32; window_tokens * hidden_dim];
    let mut actual_scratch = vec![0.0f32; window_tokens * hidden_dim];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_hidden.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                std::mem::size_of_val(actual_hidden.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_scratch.as_mut_ptr().cast::<libc::c_void>(),
                buffers.scratch_hidden_dev,
                std::mem::size_of_val(actual_scratch.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();
    unsafe {
        state.api.mem_free(attention_out_dev).unwrap();
    }

    let mut expected_hidden = Vec::with_capacity(hidden_rows.len());
    let mut expected_scratch = Vec::with_capacity(hidden_rows.len());
    for (hidden, attn) in hidden_rows
        .chunks_exact(hidden_dim)
        .zip(attention_out.chunks_exact(attn_dim))
    {
        let o_row = cpu_q4k_gemv_rows(&o_proj, hidden_dim, blocks_per_row, attn);
        let mut residual = hidden.to_vec();
        for (dst, add) in residual.iter_mut().zip(o_row.iter()) {
            *dst += *add;
        }
        expected_scratch.extend(cpu_rms_norm(&residual, &ffn_norm, 1.0e-5, false));
        expected_hidden.extend(residual);
    }

    assert_close_rows_abs_rel(
        "MTP verify attention residual hidden",
        &actual_hidden,
        &expected_hidden,
        0.04,
        0.01,
    );
    assert_close_rows_abs_rel(
        "MTP verify attention ffn norm",
        &actual_scratch,
        &expected_scratch,
        0.04,
        0.01,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_attention_moe_layer_chains_resident_stages() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4_f32_limit = usize::MAX;
    let hidden_dim = 256usize;
    let window_tokens = 2usize;
    let num_heads = 1usize;
    let num_kv_heads = 1usize;
    let head_dim = 256usize;
    let q_dim = num_heads * head_dim;
    let q_out_dim = q_dim * 2;
    let n_ff = 256usize;
    let n_expert = 3usize;
    let n_expert_used = 2usize;
    let blocks_per_row = hidden_dim / 256;
    let pos_start = 5usize;
    let rope_theta = 10000.0f32;
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.0078125)
        .collect::<Vec<_>>();
    let attn_norm = (0..hidden_dim)
        .map(|i| 0.8 + ((i % 13) as f32) * 0.0068359375)
        .collect::<Vec<_>>();
    let q_norm = (0..head_dim)
        .map(|i| 0.9 + ((i % 11) as f32) * 0.0048828125)
        .collect::<Vec<_>>();
    let k_norm = (0..head_dim)
        .map(|i| 0.85 + ((i % 7) as f32) * 0.005859375)
        .collect::<Vec<_>>();
    let ffn_norm = (0..hidden_dim)
        .map(|i| 0.7 + ((i % 17) as f32) * 0.0078125)
        .collect::<Vec<_>>();
    let q = make_test_q4k_weights(1, q_out_dim, blocks_per_row, 471)
        .pop()
        .unwrap();
    let k = make_test_q4k_weights(1, hidden_dim, blocks_per_row, 473)
        .pop()
        .unwrap();
    let v = make_test_q4k_weights(1, hidden_dim, blocks_per_row, 479)
        .pop()
        .unwrap();
    let o = make_test_q4k_weights(1, hidden_dim, blocks_per_row, 487)
        .pop()
        .unwrap();
    let mut router = vec![0.0_f32; n_expert * hidden_dim];
    router[0] = 1.0;
    router[hidden_dim + 1] = 1.0;
    router[2 * hidden_dim] = 0.25;
    router[2 * hidden_dim + 1] = 0.25;
    let gate = make_test_q4k_weights(n_expert, n_ff, blocks_per_row, 491);
    let up = make_test_q4k_weights(n_expert, n_ff, blocks_per_row, 499);
    let down = make_test_q4k_weights(n_expert, hidden_dim, n_ff / 256, 503);
    let gate_all = gate.concat();
    let up_all = up.concat();
    let down_all = down.concat();
    let shared_input_scale = (0..hidden_dim)
        .map(|i| ((i as f32 % 9.0) - 4.0) * 0.0107421875)
        .collect::<Vec<_>>();
    let shared_gate = make_test_q4k_weights(1, n_ff, blocks_per_row, 509)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, blocks_per_row, 521)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, hidden_dim, n_ff / 256, 523)
        .pop()
        .unwrap();
    let verify_tokens = [61_u32, 62];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let layer = Qwen35MtpDeviceVerifyAttentionMoeLayer {
        layer_index: 0,
        attn_norm: &attn_norm,
        q_q4k: &q,
        q_quant: 12,
        q_rows: q_out_dim,
        q_cols: hidden_dim,
        k_q4k: &k,
        k_quant: 12,
        k_rows: hidden_dim,
        k_cols: hidden_dim,
        v_q4k: &v,
        v_quant: 12,
        v_rows: hidden_dim,
        v_cols: hidden_dim,
        prior_k_bits: &[],
        prior_v_bits: &[],
        prior_tokens: 0,
        o_q4k: &o,
        o_quant: GGML_Q4_K,
        o_rows: hidden_dim,
        o_cols: hidden_dim,
        q_norm: &q_norm,
        k_norm: &k_norm,
        post_attn_norm: &[],
        ffn_norm: &ffn_norm,
        ffn_gate_q4k: &[],
        ffn_gate_rows: 0,
        ffn_gate_cols: 0,
        ffn_up_q4k: &[],
        ffn_up_rows: 0,
        ffn_up_cols: 0,
        ffn_down: &[],
        ffn_down_quant: 12,
        ffn_down_rows: 0,
        ffn_down_cols: 0,
        router_w: &router,
        n_expert,
        n_expert_used,
        gate_all: &gate_all,
        up_all: &up_all,
        down_all: &down_all,
        down_quant: 12,
        shared_input_scale: &shared_input_scale,
        shared_gate: &shared_gate,
        shared_up: &shared_up,
        shared_down: &shared_down,
        shared_down_quant: 12,
        n_ff,
        n_embd: hidden_dim,
    };

    let attention_kv = state
        .stage_mtp_verify_qwen35_attention_moe_layer_q4k_with_kv_state(
            &buffers, &layer, head_dim, true, rope_theta, pos_start, 1.0e-5,
        )
        .expect("stage attention MoE layer");

    let mut actual = vec![0.0f32; hidden_rows.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut q_raw_full = Vec::with_capacity(window_tokens * q_out_dim);
    let mut k_raw = Vec::with_capacity(window_tokens * hidden_dim);
    let mut v_raw = Vec::with_capacity(window_tokens * hidden_dim);
    for hidden in hidden_rows.chunks_exact(hidden_dim) {
        let normed = cpu_rms_norm(hidden, &attn_norm, 1.0e-5, false);
        q_raw_full.extend(cpu_q4k_gemv_rows(&q, q_out_dim, blocks_per_row, &normed));
        k_raw.extend(cpu_q4k_gemv_rows(&k, hidden_dim, blocks_per_row, &normed));
        v_raw.extend(cpu_q4k_gemv_rows(&v, hidden_dim, blocks_per_row, &normed));
    }
    let mut q_raw = vec![0.0f32; window_tokens * q_dim];
    let mut gate_raw = vec![0.0f32; window_tokens * q_dim];
    for t in 0..window_tokens {
        for h in 0..num_heads {
            let src = t * q_out_dim + h * head_dim * 2;
            let dst = t * q_dim + h * head_dim;
            q_raw[dst..dst + head_dim].copy_from_slice(&q_raw_full[src..src + head_dim]);
            gate_raw[dst..dst + head_dim]
                .copy_from_slice(&q_raw_full[src + head_dim..src + head_dim * 2]);
        }
    }
    let q_post = cpu_qk_norm_rope_neox(
        &q_raw,
        &q_norm,
        window_tokens,
        num_heads,
        head_dim,
        pos_start,
        1.0e-5,
        rope_theta,
        None,
        false,
    );
    let k_post = cpu_qk_norm_rope_neox(
        &k_raw,
        &k_norm,
        window_tokens,
        num_kv_heads,
        head_dim,
        pos_start,
        1.0e-5,
        rope_theta,
        None,
        false,
    );
    let k_f16 = k_post
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    let v_f16 = v_raw
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect::<Vec<_>>();
    assert_eq!(attention_kv.layer_idx, 0);
    assert_eq!(attention_kv.window_tokens, window_tokens);
    assert_eq!(attention_kv.kv_rows, hidden_dim);
    assert_eq!(attention_kv.k_bits.len(), k_f16.len());
    assert_eq!(attention_kv.v_bits.len(), v_f16.len());
    let attention_k = attention_kv
        .k_bits
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect::<Vec<_>>();
    let attention_v = attention_kv
        .v_bits
        .iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect::<Vec<_>>();
    assert_close_rows_abs_rel(
        "MTP verify attention layer captured K bits",
        &attention_k,
        &k_post,
        0.05,
        0.01,
    );
    assert_close_rows_abs_rel(
        "MTP verify attention layer captured V bits",
        &attention_v,
        &v_raw,
        0.05,
        0.01,
    );
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; window_tokens * hidden_dim];
    for t in 0..window_tokens {
        let q_row = &q_post[t * hidden_dim..(t + 1) * hidden_dim];
        let mut scores = Vec::with_capacity(t + 1);
        for j in 0..=t {
            let k_off = j * hidden_dim;
            let dot = q_row
                .iter()
                .zip(k_f16[k_off..k_off + hidden_dim].iter())
                .map(|(a, b)| a * half::f16::from_bits(*b).to_f32())
                .sum::<f32>()
                * scale;
            scores.push((j, dot));
        }
        let max_score = scores
            .iter()
            .map(|(_, score)| *score)
            .fold(f32::NEG_INFINITY, f32::max);
        let denom = scores
            .iter()
            .map(|(_, score)| (*score - max_score).exp())
            .sum::<f32>();
        for (j, score) in scores {
            let p = (score - max_score).exp() / denom;
            let v_off = j * hidden_dim;
            let out_off = t * hidden_dim;
            for d in 0..hidden_dim {
                attn_out[out_off + d] += p * half::f16::from_bits(v_f16[v_off + d]).to_f32();
            }
        }
    }
    for (out, gate) in attn_out.iter_mut().zip(gate_raw.iter()) {
        *out *= 1.0 / (1.0 + (-gate).exp());
    }

    let mut expected = Vec::with_capacity(hidden_rows.len());
    for (hidden, attn) in hidden_rows
        .chunks_exact(hidden_dim)
        .zip(attn_out.chunks_exact(hidden_dim))
    {
        let o_row = cpu_q4k_gemv_rows(&o, hidden_dim, blocks_per_row, attn);
        let mut residual = hidden.to_vec();
        for (dst, add) in residual.iter_mut().zip(o_row.iter()) {
            *dst += *add;
        }
        let moe_input = cpu_rms_norm(&residual, &ffn_norm, 1.0e-5, false);
        let mut logits = vec![0.0_f32; n_expert];
        for expert in 0..n_expert {
            let w = &router[expert * hidden_dim..(expert + 1) * hidden_dim];
            logits[expert] = moe_input
                .iter()
                .zip(w.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
        }
        let mut ranked = (0..n_expert).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let selected = &ranked[..n_expert_used];
        let max_selected = selected
            .iter()
            .map(|&expert| logits[expert])
            .fold(f32::NEG_INFINITY, f32::max);
        let selected_sum = selected
            .iter()
            .map(|&expert| (logits[expert] - max_selected).exp())
            .sum::<f32>();
        let gate_refs = selected
            .iter()
            .map(|&expert| gate[expert].as_slice())
            .collect::<Vec<_>>();
        let up_refs = selected
            .iter()
            .map(|&expert| up[expert].as_slice())
            .collect::<Vec<_>>();
        let down_refs = selected
            .iter()
            .map(|&expert| down[expert].as_slice())
            .collect::<Vec<_>>();
        let routes = selected
            .iter()
            .map(|&expert| (logits[expert] - max_selected).exp() / selected_sum)
            .collect::<Vec<_>>();
        let sparse = cpu_qwen35_sparse_q4k_reference(
            &gate_refs, &up_refs, &down_refs, &routes, n_ff, hidden_dim, &moe_input,
        );
        let shared_gate_dot = moe_input
            .iter()
            .zip(shared_input_scale.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let shared_route = 1.0 / (1.0 + (-shared_gate_dot).exp());
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, blocks_per_row, &moe_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, blocks_per_row, &moe_input);
        let shared_hidden = shared_gate_out
            .iter()
            .zip(shared_up_out.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect::<Vec<_>>();
        let shared = cpu_q4k_gemv_rows(&shared_down, hidden_dim, n_ff / 256, &shared_hidden);
        for row in 0..hidden_dim {
            residual[row] += sparse[row] + shared[row] * shared_route;
        }
        expected.extend(residual);
    }

    assert_close_rows_abs_rel(
        "MTP verify attention MoE layer",
        &actual,
        &expected,
        0.75,
        0.04,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_conv1d_silu_uses_device_qkv_rows() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 3usize;
    let channels = 32usize;
    let gate_rows = 16usize;
    let head_rows = 4usize;
    let kernel_size = 4usize;
    let blocks_per_row = hidden_dim / 256;
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.009765625)
        .collect::<Vec<_>>();
    let attn_norm = vec![1.0f32; hidden_dim];
    let qkv = make_test_q4k_weights(1, channels, blocks_per_row, 373)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, gate_rows, blocks_per_row, 379)
        .pop()
        .unwrap();
    let alpha = make_test_q4k_weights(1, head_rows, blocks_per_row, 383)
        .pop()
        .unwrap();
    let beta = make_test_q4k_weights(1, head_rows, blocks_per_row, 389)
        .pop()
        .unwrap();
    let conv_state = (0..(kernel_size - 1) * channels)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.017578125)
        .collect::<Vec<_>>();
    let conv_kernel = (0..kernel_size * channels)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.021484375)
        .collect::<Vec<_>>();
    let verify_tokens = [17_u32, 18, 19];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let projection_buffers = state
        .stage_mtp_verify_gdn_input_projections_q4k(
            &buffers,
            mtp_verify::Qwen35MtpGdnProjectionRequest {
                attn_norm: &attn_norm,
                qkv_q4k: &qkv,
                qkv_quant: 12,
                qkv_rows: channels,
                qkv_cols: hidden_dim,
                gate_q4k: &gate,
                gate_rows,
                gate_cols: hidden_dim,
                alpha_q4k: &alpha,
                alpha_f32: &[],
                alpha_quant: GGML_Q4_K,
                alpha_rows: head_rows,
                alpha_cols: hidden_dim,
                beta_q4k: &beta,
                beta_f32: &[],
                beta_quant: GGML_Q4_K,
                beta_rows: head_rows,
                beta_cols: hidden_dim,
                norm_eps: 1.0e-5,
            },
        )
        .expect("stage GDN projections");

    let conv_buffers = state
        .stage_mtp_verify_gdn_conv1d_silu(
            &projection_buffers,
            &conv_state,
            &conv_kernel,
            kernel_size,
        )
        .expect("stage GDN conv1d silu");

    let mut actual = vec![0.0f32; window_tokens * channels];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                conv_buffers.conv_out_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut qkv_rows = Vec::with_capacity(window_tokens * channels);
    for hidden in hidden_rows.chunks_exact(hidden_dim) {
        let normed = cpu_rms_norm(hidden, &attn_norm, 1.0e-5, false);
        qkv_rows.extend(cpu_q4k_gemv_rows(&qkv, channels, blocks_per_row, &normed));
    }
    let mut conv_input = conv_state.clone();
    conv_input.extend_from_slice(&qkv_rows);
    let mut expected = vec![0.0f32; window_tokens * channels];
    for token_idx in 0..window_tokens {
        for channel_idx in 0..channels {
            let mut sum = 0.0f32;
            for k in 0..kernel_size {
                sum += conv_input[(token_idx + k) * channels + channel_idx]
                    * conv_kernel[k * channels + channel_idx];
            }
            expected[token_idx * channels + channel_idx] = sum / (1.0 + (-sum).exp());
        }
    }

    assert_close_rows("MTP verify GDN conv1d silu", &actual, &expected, 2.0e-5);
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_conv_prefix_state_reads_device_conv_input() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let window_tokens = 3usize;
    let channels = 5usize;
    let kernel_size = 4usize;
    let conv_state = (0..(kernel_size - 1) * channels)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.03125)
        .collect::<Vec<_>>();
    let qkv_rows = (0..window_tokens * channels)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.046875)
        .collect::<Vec<_>>();
    let conv_buffers = state
        .ensure_mtp_verify_gdn_conv_buffers(window_tokens, channels, kernel_size)
        .expect("allocate conv buffers");
    let mut conv_input = conv_state.clone();
    conv_input.extend_from_slice(&qkv_rows);
    unsafe {
        state
            .api
            .memcpy_htod_async(
                conv_buffers.conv_input_dev,
                conv_input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(conv_input.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let prefix_tokens = 2usize;
    let actual = state
        .stage_mtp_verify_gdn_conv_prefix_state(&conv_buffers, prefix_tokens)
        .expect("copy prefix conv state");
    let conv_state_len = (kernel_size - 1) * channels;
    let expected =
        conv_input[prefix_tokens * channels..prefix_tokens * channels + conv_state_len].to_vec();

    assert_eq!(actual, expected);

    let deferred_prefix = state
        .stage_mtp_verify_gdn_conv_prefix_state_deferred(&conv_buffers, prefix_tokens)
        .expect("queue deferred prefix conv state");
    let deferred_final = state
        .stage_mtp_verify_gdn_conv_final_state_deferred(&conv_buffers)
        .expect("queue deferred final conv state");
    state.stream_synchronize().unwrap();
    let final_expected =
        conv_input[window_tokens * channels..window_tokens * channels + conv_state_len].to_vec();
    assert_eq!(deferred_prefix, expected);
    assert_eq!(deferred_final, final_expected);
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_delta_inputs_match_cpu_reference() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let window_tokens = 2usize;
    let num_k_heads = 2usize;
    let num_v_heads = 4usize;
    let head_k_dim = 4usize;
    let head_v_dim = 3usize;
    let q_dim = num_k_heads * head_k_dim;
    let k_dim = num_k_heads * head_k_dim;
    let v_dim = num_v_heads * head_v_dim;
    let conv_channels = q_dim + k_dim + v_dim;
    let kernel_size = 2usize;
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, 64, 0).unwrap();
    let projection_buffers = state
        .ensure_mtp_verify_gdn_projection_buffers(
            &plan,
            conv_channels,
            v_dim,
            num_v_heads,
            num_v_heads,
        )
        .expect("allocate projection buffers");
    let conv_buffers = state
        .ensure_mtp_verify_gdn_conv_buffers(window_tokens, conv_channels, kernel_size)
        .expect("allocate conv buffers");
    let conv_out = (0..window_tokens * conv_channels)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.03125)
        .collect::<Vec<_>>();
    let alpha = (0..window_tokens * num_v_heads)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.125)
        .collect::<Vec<_>>();
    let beta = (0..window_tokens * num_v_heads)
        .map(|i| ((i as f32 % 13.0) - 6.0) * 0.09375)
        .collect::<Vec<_>>();
    let dt_bias = [-0.25f32, 0.0, 0.25, 0.5];
    let ssm_a = [-0.75f32, -0.5, -0.25, -0.125];
    unsafe {
        state
            .api
            .memcpy_htod_async(
                conv_buffers.conv_out_dev,
                conv_out.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(conv_out.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                projection_buffers.alpha_dev,
                alpha.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(alpha.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                projection_buffers.beta_dev,
                beta.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(beta.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let delta_buffers = state
        .stage_mtp_verify_gdn_delta_inputs(
            &conv_buffers,
            &projection_buffers,
            &dt_bias,
            &ssm_a,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            1.0e-5,
        )
        .expect("stage delta inputs");

    let mut actual_q = vec![0.0f32; window_tokens * num_v_heads * head_k_dim];
    let mut actual_k = vec![0.0f32; actual_q.len()];
    let mut actual_v = vec![0.0f32; window_tokens * num_v_heads * head_v_dim];
    let mut actual_gate = vec![0.0f32; window_tokens * num_v_heads];
    let mut actual_beta = vec![0.0f32; window_tokens * num_v_heads];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_q.as_mut_ptr().cast::<libc::c_void>(),
                delta_buffers.q_dev,
                std::mem::size_of_val(actual_q.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_k.as_mut_ptr().cast::<libc::c_void>(),
                delta_buffers.k_dev,
                std::mem::size_of_val(actual_k.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_v.as_mut_ptr().cast::<libc::c_void>(),
                delta_buffers.v_dev,
                std::mem::size_of_val(actual_v.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_gate.as_mut_ptr().cast::<libc::c_void>(),
                delta_buffers.gate_dev,
                std::mem::size_of_val(actual_gate.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_beta.as_mut_ptr().cast::<libc::c_void>(),
                delta_buffers.beta_dev,
                std::mem::size_of_val(actual_beta.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let scale = 1.0 / (head_k_dim as f32).sqrt();
    let mut expected_q = vec![0.0f32; actual_q.len()];
    let mut expected_k = vec![0.0f32; actual_k.len()];
    let mut expected_v = vec![0.0f32; actual_v.len()];
    let mut expected_gate = vec![0.0f32; actual_gate.len()];
    let mut expected_beta = vec![0.0f32; actual_beta.len()];
    for token_idx in 0..window_tokens {
        let row = &conv_out[token_idx * conv_channels..(token_idx + 1) * conv_channels];
        for k_head in 0..num_k_heads {
            let q_src = &row[k_head * head_k_dim..(k_head + 1) * head_k_dim];
            let k_src = &row[q_dim + k_head * head_k_dim..q_dim + (k_head + 1) * head_k_dim];
            let q_inv = 1.0 / (q_src.iter().map(|v| v * v).sum::<f32>() + 1.0e-5).sqrt();
            let k_inv = 1.0 / (k_src.iter().map(|v| v * v).sum::<f32>() + 1.0e-5).sqrt();
            for v_head in (k_head..num_v_heads).step_by(num_k_heads) {
                let out = (token_idx * num_v_heads + v_head) * head_k_dim;
                for dim in 0..head_k_dim {
                    expected_q[out + dim] = q_src[dim] * q_inv * scale;
                    expected_k[out + dim] = k_src[dim] * k_inv;
                }
            }
        }
        for v_head in 0..num_v_heads {
            let v_src = q_dim + k_dim + v_head * head_v_dim;
            let v_out = (token_idx * num_v_heads + v_head) * head_v_dim;
            expected_v[v_out..v_out + head_v_dim].copy_from_slice(&row[v_src..v_src + head_v_dim]);
            let gate_idx = token_idx * num_v_heads + v_head;
            let biased = alpha[gate_idx] + dt_bias[v_head];
            expected_gate[gate_idx] = (1.0 + biased.exp()).ln() * ssm_a[v_head];
            expected_beta[gate_idx] = 1.0 / (1.0 + (-beta[gate_idx]).exp());
        }
    }

    assert_close_rows("MTP verify GDN delta q", &actual_q, &expected_q, 2.0e-5);
    assert_close_rows("MTP verify GDN delta k", &actual_k, &expected_k, 2.0e-5);
    assert_close_rows("MTP verify GDN delta v", &actual_v, &expected_v, 0.0);
    assert_close_rows(
        "MTP verify GDN delta gate",
        &actual_gate,
        &expected_gate,
        2.0e-5,
    );
    assert_close_rows(
        "MTP verify GDN delta beta",
        &actual_beta,
        &expected_beta,
        2.0e-5,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_delta_scan_uses_device_inputs() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let seq_len = 3usize;
    let num_heads = 2usize;
    let head_k_dim = 4usize;
    let head_v_dim = 3usize;
    let delta_buffers = state
        .ensure_mtp_verify_gdn_delta_input_buffers(seq_len, num_heads, head_k_dim, head_v_dim)
        .expect("allocate delta input buffers");
    let q = (0..seq_len * num_heads * head_k_dim)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.03125)
        .collect::<Vec<_>>();
    let k = (0..q.len())
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.02734375)
        .collect::<Vec<_>>();
    let v = (0..seq_len * num_heads * head_v_dim)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.0234375)
        .collect::<Vec<_>>();
    let gate = (0..seq_len * num_heads)
        .map(|i| -0.125 - (i as f32 % 5.0) * 0.03125)
        .collect::<Vec<_>>();
    let beta = (0..seq_len * num_heads)
        .map(|i| 0.35 + (i as f32 % 7.0) * 0.041015625)
        .collect::<Vec<_>>();
    unsafe {
        state
            .api
            .memcpy_htod_async(
                delta_buffers.q_dev,
                q.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(q.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                delta_buffers.k_dev,
                k.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(k.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                delta_buffers.v_dev,
                v.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(v.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                delta_buffers.gate_dev,
                gate.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(gate.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                delta_buffers.beta_dev,
                beta.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(beta.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let mut delta_state = (0..num_heads * head_v_dim * head_k_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.015625)
        .collect::<Vec<_>>();
    let mut expected_state = delta_state.clone();
    let expected = cpu_delta_net_prefill_reference(
        &mut expected_state,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
    );

    let scan_buffers = state
        .stage_mtp_verify_gdn_delta_scan(&delta_buffers, &mut delta_state, true)
        .expect("stage delta scan");

    let mut actual = vec![0.0f32; expected.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                scan_buffers.output_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    assert_close_rows("MTP verify GDN delta scan", &actual, &expected, 2.0e-5);
    assert_close_rows(
        "MTP verify GDN delta state",
        &delta_state,
        &expected_state,
        2.0e-5,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_delta_scan_captures_prefix_snapshots() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let seq_len = 3usize;
    let num_heads = 2usize;
    let head_k_dim = 4usize;
    let head_v_dim = 3usize;
    let delta_buffers = state
        .ensure_mtp_verify_gdn_delta_input_buffers(seq_len, num_heads, head_k_dim, head_v_dim)
        .expect("allocate delta input buffers");
    let q = (0..seq_len * num_heads * head_k_dim)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.03125)
        .collect::<Vec<_>>();
    let k = (0..q.len())
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.02734375)
        .collect::<Vec<_>>();
    let v = (0..seq_len * num_heads * head_v_dim)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.0234375)
        .collect::<Vec<_>>();
    let gate = (0..seq_len * num_heads)
        .map(|i| -0.125 - (i as f32 % 5.0) * 0.03125)
        .collect::<Vec<_>>();
    let beta = (0..seq_len * num_heads)
        .map(|i| 0.35 + (i as f32 % 7.0) * 0.041015625)
        .collect::<Vec<_>>();
    unsafe {
        state
            .api
            .memcpy_htod_async(
                delta_buffers.q_dev,
                q.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(q.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                delta_buffers.k_dev,
                k.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(k.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                delta_buffers.v_dev,
                v.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(v.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                delta_buffers.gate_dev,
                gate.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(gate.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                delta_buffers.beta_dev,
                beta.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(beta.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let mut delta_state = (0..num_heads * head_v_dim * head_k_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.015625)
        .collect::<Vec<_>>();
    let mut expected_prefix1 = delta_state.clone();
    cpu_delta_net_prefill_reference(
        &mut expected_prefix1,
        &q[..num_heads * head_k_dim],
        &k[..num_heads * head_k_dim],
        &v[..num_heads * head_v_dim],
        &gate[..num_heads],
        &beta[..num_heads],
        1,
        num_heads,
        head_k_dim,
        head_v_dim,
    );
    let mut expected_prefix2 = delta_state.clone();
    cpu_delta_net_prefill_reference(
        &mut expected_prefix2,
        &q[..2 * num_heads * head_k_dim],
        &k[..2 * num_heads * head_k_dim],
        &v[..2 * num_heads * head_v_dim],
        &gate[..2 * num_heads],
        &beta[..2 * num_heads],
        2,
        num_heads,
        head_k_dim,
        head_v_dim,
    );
    let mut expected_final = delta_state.clone();
    let expected_output = cpu_delta_net_prefill_reference(
        &mut expected_final,
        &q,
        &k,
        &v,
        &gate,
        &beta,
        seq_len,
        num_heads,
        head_k_dim,
        head_v_dim,
    );

    let (scan_buffers, snapshots) = state
        .stage_mtp_verify_gdn_delta_scan_snapshots(&delta_buffers, &mut delta_state, true, &[1, 2])
        .expect("stage delta scan snapshots");
    assert_eq!(snapshots.len(), 2);

    let mut actual_output = vec![0.0f32; expected_output.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_output.as_mut_ptr().cast::<libc::c_void>(),
                scan_buffers.output_dev,
                std::mem::size_of_val(actual_output.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();
    assert_close_rows(
        "MTP verify GDN delta snapshot output",
        &actual_output,
        &expected_output,
        2.0e-5,
    );
    assert_close_rows(
        "MTP verify GDN delta snapshot final state",
        &delta_state,
        &expected_final,
        2.0e-5,
    );

    state
        .restore_resident_delta_state(&mut delta_state, &snapshots[0])
        .expect("restore prefix 1");
    state
        .sync_resident_delta_state(&mut delta_state)
        .expect("sync prefix 1");
    assert_close_rows(
        "MTP verify GDN delta prefix 1 state",
        &delta_state,
        &expected_prefix1,
        2.0e-5,
    );
    state
        .restore_resident_delta_state(&mut delta_state, &snapshots[1])
        .expect("restore prefix 2");
    state
        .sync_resident_delta_state(&mut delta_state)
        .expect("sync prefix 2");
    assert_close_rows(
        "MTP verify GDN delta prefix 2 state",
        &delta_state,
        &expected_prefix2,
        2.0e-5,
    );

    for snapshot in snapshots {
        state
            .free_delta_state_snapshot(snapshot)
            .expect("free delta snapshot");
    }
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_ssm_out_projects_device_delta_output() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let seq_len = 2usize;
    let num_heads = 2usize;
    let head_v_dim = 128usize;
    let d_inner = num_heads * head_v_dim;
    let hidden_rows = 12usize;
    let delta_inputs = state
        .ensure_mtp_verify_gdn_delta_input_buffers(seq_len, num_heads, 4, head_v_dim)
        .expect("allocate delta buffers");
    let scan_buffers = state
        .ensure_mtp_verify_gdn_delta_scan_buffers(&delta_inputs)
        .expect("allocate delta scan buffers");
    let plan = qwen35_mtp_verify_buffer_plan(seq_len, 512, 0).unwrap();
    let projection_buffers = state
        .ensure_mtp_verify_gdn_projection_buffers(&plan, d_inner, d_inner, num_heads, num_heads)
        .expect("allocate projection buffers");
    let delta_out = (0..seq_len * d_inner)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.015625)
        .collect::<Vec<_>>();
    let z = (0..seq_len * d_inner)
        .map(|i| ((i as f32 % 41.0) - 20.0) * 0.01953125)
        .collect::<Vec<_>>();
    let norm = (0..head_v_dim)
        .map(|i| 0.75 + (i as f32 % 9.0) * 0.03125)
        .collect::<Vec<_>>();
    let ssm_out = make_test_q4k_weights(1, hidden_rows, d_inner / 256, 421)
        .pop()
        .unwrap();
    unsafe {
        state
            .api
            .memcpy_htod_async(
                scan_buffers.output_dev,
                delta_out.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(delta_out.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                projection_buffers.gate_dev,
                z.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(z.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let ssm_buffers = state
        .stage_mtp_verify_gdn_ssm_out_q4k(
            &scan_buffers,
            &projection_buffers,
            &norm,
            &ssm_out,
            GGML_Q4_K,
            hidden_rows,
            d_inner,
            1.0e-5,
        )
        .expect("stage ssm out");

    let mut actual = vec![0.0f32; seq_len * hidden_rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                ssm_buffers.ssm_out_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut gated = vec![0.0f32; seq_len * d_inner];
    for row_idx in 0..seq_len * num_heads {
        let start = row_idx * head_v_dim;
        let row = &delta_out[start..start + head_v_dim];
        let inv =
            1.0 / (row.iter().map(|v| v * v).sum::<f32>() / head_v_dim as f32 + 1.0e-5).sqrt();
        for dim in 0..head_v_dim {
            let idx = start + dim;
            let z_value = z[idx];
            gated[idx] = row[dim] * inv * norm[dim] * (z_value / (1.0 + (-z_value).exp()));
        }
    }
    let mut expected = Vec::with_capacity(seq_len * hidden_rows);
    for token_idx in 0..seq_len {
        let input = &gated[token_idx * d_inner..(token_idx + 1) * d_inner];
        expected.extend(cpu_q4k_gemv_rows(
            &ssm_out,
            hidden_rows,
            d_inner / 256,
            input,
        ));
    }

    assert_close_rows("MTP verify GDN ssm_out", &actual, &expected, 0.08);
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_ssm_out_accepts_q8_0_projection() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let seq_len = 2usize;
    let num_heads = 2usize;
    let head_v_dim = 128usize;
    let d_inner = num_heads * head_v_dim;
    let hidden_rows = 10usize;
    let delta_inputs = state
        .ensure_mtp_verify_gdn_delta_input_buffers(seq_len, num_heads, 4, head_v_dim)
        .expect("allocate delta buffers");
    let scan_buffers = state
        .ensure_mtp_verify_gdn_delta_scan_buffers(&delta_inputs)
        .expect("allocate delta scan buffers");
    let plan = qwen35_mtp_verify_buffer_plan(seq_len, 512, 0).unwrap();
    let projection_buffers = state
        .ensure_mtp_verify_gdn_projection_buffers(&plan, d_inner, d_inner, num_heads, num_heads)
        .expect("allocate projection buffers");
    let delta_out = (0..seq_len * d_inner)
        .map(|i| ((i as f32 % 43.0) - 21.0) * 0.01171875)
        .collect::<Vec<_>>();
    let z = (0..seq_len * d_inner)
        .map(|i| ((i as f32 % 47.0) - 23.0) * 0.013671875)
        .collect::<Vec<_>>();
    let norm = (0..head_v_dim)
        .map(|i| 0.625 + (i as f32 % 7.0) * 0.02734375)
        .collect::<Vec<_>>();
    let ssm_out = make_test_q8_0_weights(hidden_rows, d_inner, 431);
    unsafe {
        state
            .api
            .memcpy_htod_async(
                scan_buffers.output_dev,
                delta_out.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(delta_out.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                projection_buffers.gate_dev,
                z.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(z.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let ssm_buffers = state
        .stage_mtp_verify_gdn_ssm_out_q4k(
            &scan_buffers,
            &projection_buffers,
            &norm,
            &ssm_out,
            GGML_Q8_0,
            hidden_rows,
            d_inner,
            1.0e-5,
        )
        .expect("stage Q8_0 ssm out");

    let mut actual = vec![0.0f32; seq_len * hidden_rows];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                ssm_buffers.ssm_out_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut gated = vec![0.0f32; seq_len * d_inner];
    for row_idx in 0..seq_len * num_heads {
        let start = row_idx * head_v_dim;
        let row = &delta_out[start..start + head_v_dim];
        let inv =
            1.0 / (row.iter().map(|v| v * v).sum::<f32>() / head_v_dim as f32 + 1.0e-5).sqrt();
        for dim in 0..head_v_dim {
            let idx = start + dim;
            let z_value = z[idx];
            gated[idx] = row[dim] * inv * norm[dim] * (z_value / (1.0 + (-z_value).exp()));
        }
    }
    let mut expected = Vec::with_capacity(seq_len * hidden_rows);
    for token_idx in 0..seq_len {
        let input = &gated[token_idx * d_inner..(token_idx + 1) * d_inner];
        expected.extend(cpu_q8_0_rows(&ssm_out, hidden_rows, d_inner, input));
    }

    assert_close_rows("MTP verify GDN ssm_out Q8_0", &actual, &expected, 0.03);
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_residual_post_norm_stays_device_resident() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let window_tokens = 2usize;
    let hidden_dim = 16usize;
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let verify_tokens = [7_u32, 8];
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify buffers");
    let ssm_buffers = state
        .ensure_mtp_verify_gdn_ssm_out_buffers(window_tokens, 8, hidden_dim)
        .expect("allocate ssm output buffers");
    let hidden = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.03125)
        .collect::<Vec<_>>();
    let ssm_out = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.02734375)
        .collect::<Vec<_>>();
    let post_norm = (0..hidden_dim)
        .map(|i| 0.5 + (i as f32 % 7.0) * 0.0625)
        .collect::<Vec<_>>();
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                ssm_buffers.ssm_out_dev,
                ssm_out.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(ssm_out.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    state
        .stage_mtp_verify_gdn_residual_post_norm(&buffers, &ssm_buffers, &post_norm, 1.0e-5)
        .expect("stage residual add and post norm");

    let mut actual_hidden = vec![0.0f32; hidden.len()];
    let mut actual_normed = vec![0.0f32; hidden.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_hidden.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                std::mem::size_of_val(actual_hidden.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_normed.as_mut_ptr().cast::<libc::c_void>(),
                buffers.scratch_hidden_dev,
                std::mem::size_of_val(actual_normed.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let expected_hidden = hidden
        .iter()
        .zip(ssm_out.iter())
        .map(|(h, s)| h + s)
        .collect::<Vec<_>>();
    let mut expected_normed = Vec::with_capacity(expected_hidden.len());
    for row in expected_hidden.chunks_exact(hidden_dim) {
        expected_normed.extend(cpu_rms_norm(row, &post_norm, 1.0e-5, false));
    }

    assert_close_rows(
        "MTP verify GDN residual hidden",
        &actual_hidden,
        &expected_hidden,
        0.0,
    );
    assert_close_rows(
        "MTP verify GDN residual post norm",
        &actual_normed,
        &expected_normed,
        2.0e-5,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_sparse_moe_adds_to_hidden_rows_from_scratch() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let window_tokens = 2usize;
    let hidden_dim = 256usize;
    let n_ff = 256usize;
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let verify_tokens = [3_u32, 4];
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify buffers");
    let hidden_after_ssm = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.01953125)
        .collect::<Vec<_>>();
    let moe_input = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.015625)
        .collect::<Vec<_>>();
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_after_ssm.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_after_ssm.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                buffers.scratch_hidden_dev,
                moe_input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(moe_input.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    let gate = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 431);
    let up = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 433);
    let down = make_test_q4k_weights(1, hidden_dim, n_ff / 256, 439);
    let gate_refs = vec![gate[0].as_slice(), gate[0].as_slice()];
    let up_refs = vec![up[0].as_slice(), up[0].as_slice()];
    let down_refs = vec![down[0].as_slice(), down[0].as_slice()];
    let route = vec![0.625f32, 0.375f32];
    let token_ids = vec![0_u32, 1];

    state
        .stage_mtp_verify_gdn_sparse_moe_by_token_q4k(
            &buffers, &gate_refs, &up_refs, &down_refs, &route, &token_ids, 12, n_ff, hidden_dim,
        )
        .expect("stage sparse MoE");

    let mut actual = vec![0.0f32; hidden_after_ssm.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected = hidden_after_ssm.clone();
    for token in 0..window_tokens {
        let token_input = &moe_input[token * hidden_dim..(token + 1) * hidden_dim];
        let moe = cpu_qwen35_sparse_q4k_reference(
            &[gate[0].as_slice()],
            &[up[0].as_slice()],
            &[down[0].as_slice()],
            &[route[token]],
            n_ff,
            hidden_dim,
            token_input,
        );
        for row in 0..hidden_dim {
            expected[token * hidden_dim + row] += moe[row];
        }
    }

    assert_close_rows_abs_rel(
        "MTP verify GDN sparse MoE residual",
        &actual,
        &expected,
        0.2,
        0.02,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_router_topk_matches_softmax_reference() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let window_tokens = 2usize;
    let hidden_dim = 4usize;
    let n_expert = 5usize;
    let n_expert_used = 2usize;
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let verify_tokens = [7_u32, 8];
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify buffers");
    let hidden = [1.0_f32, -0.5, 0.25, 0.75, -0.25, 0.5, 1.5, -1.0];
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.scratch_hidden_dev,
                hidden.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let router = [
        0.5_f32, -0.25, 0.0, 0.125, -0.75, 0.5, 0.25, -0.125, 0.25, 0.375, -0.5, 0.0, -0.125,
        -0.25, 0.75, 0.5, 0.0, 0.125, 0.5, -0.375,
    ];

    let route_buffers = state
        .stage_mtp_verify_qwen35_router_topk(&buffers, &router, n_expert, n_expert_used)
        .expect("stage router topk");

    let slots = window_tokens * n_expert_used;
    let mut actual_experts = vec![0_u32; slots];
    let mut actual_weights = vec![0.0_f32; slots];
    let mut actual_tokens = vec![0_u32; slots];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual_experts.as_mut_ptr().cast::<libc::c_void>(),
                route_buffers.expert_ids_dev,
                std::mem::size_of_val(actual_experts.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_weights.as_mut_ptr().cast::<libc::c_void>(),
                route_buffers.route_weights_dev,
                std::mem::size_of_val(actual_weights.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_dtoh_async(
                actual_tokens.as_mut_ptr().cast::<libc::c_void>(),
                route_buffers.token_ids_dev,
                std::mem::size_of_val(actual_tokens.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected_experts = Vec::with_capacity(slots);
    let mut expected_weights = Vec::with_capacity(slots);
    let mut expected_tokens = Vec::with_capacity(slots);
    for token in 0..window_tokens {
        let h = &hidden[token * hidden_dim..(token + 1) * hidden_dim];
        let mut logits = vec![0.0_f32; n_expert];
        for expert in 0..n_expert {
            let w = &router[expert * hidden_dim..(expert + 1) * hidden_dim];
            logits[expert] = h.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>();
        }
        let mut ranked = (0..n_expert).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let selected = &ranked[..n_expert_used];
        let max_selected = selected
            .iter()
            .map(|&expert| logits[expert])
            .fold(f32::NEG_INFINITY, f32::max);
        let selected_sum = selected
            .iter()
            .map(|&expert| (logits[expert] - max_selected).exp())
            .sum::<f32>();
        for &expert in selected {
            expected_experts.push(expert as u32);
            expected_weights.push((logits[expert] - max_selected).exp() / selected_sum);
            expected_tokens.push(token as u32);
        }
    }

    assert_eq!(actual_experts, expected_experts);
    assert_eq!(actual_tokens, expected_tokens);
    assert_close_rows(
        "MTP verify router route weights",
        &actual_weights,
        &expected_weights,
        1.0e-6,
    );
    assert_eq!(route_buffers.window_tokens, window_tokens);
    assert_eq!(route_buffers.n_expert, n_expert);
    assert_eq!(route_buffers.n_expert_used, n_expert_used);
}

#[test]
fn cuda_qwen35_mtp_verify_sparse_moe_consumes_device_router_slots() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let window_tokens = 2usize;
    let hidden_dim = 256usize;
    let n_ff = 256usize;
    let n_expert = 3usize;
    let n_expert_used = 2usize;
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let verify_tokens = [11_u32, 12];
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify buffers");
    let hidden_after_ssm = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 17.0) - 8.0) * 0.01171875)
        .collect::<Vec<_>>();
    let mut moe_input = vec![0.0_f32; window_tokens * hidden_dim];
    moe_input[0] = 1.5;
    moe_input[1] = 0.25;
    moe_input[2] = 0.75;
    moe_input[hidden_dim] = 0.125;
    moe_input[hidden_dim + 1] = 1.75;
    moe_input[hidden_dim + 2] = 0.5;
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_after_ssm.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_after_ssm.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                buffers.scratch_hidden_dev,
                moe_input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(moe_input.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let mut router = vec![0.0_f32; n_expert * hidden_dim];
    router[2] = 0.4;
    router[hidden_dim] = 1.0;
    router[2 * hidden_dim + 1] = 1.0;
    let route_buffers = state
        .stage_mtp_verify_qwen35_router_topk(&buffers, &router, n_expert, n_expert_used)
        .expect("stage router topk");
    let gate = make_test_q4k_weights(n_expert, n_ff, hidden_dim / 256, 541);
    let up = make_test_q4k_weights(n_expert, n_ff, hidden_dim / 256, 547);
    let down = make_test_q4k_weights(n_expert, hidden_dim, n_ff / 256, 557);
    let gate_all = gate.concat();
    let up_all = up.concat();
    let down_all = down.concat();

    state
        .stage_mtp_verify_gdn_sparse_moe_full_layer_from_router_q4k(
            &buffers,
            &route_buffers,
            &gate_all,
            &up_all,
            &down_all,
            12,
            n_expert,
            n_ff,
            hidden_dim,
        )
        .expect("stage sparse MoE from device router");

    let mut actual = vec![0.0f32; hidden_after_ssm.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected = hidden_after_ssm.clone();
    for token in 0..window_tokens {
        let h = &moe_input[token * hidden_dim..(token + 1) * hidden_dim];
        let mut logits = vec![0.0_f32; n_expert];
        for expert in 0..n_expert {
            let w = &router[expert * hidden_dim..(expert + 1) * hidden_dim];
            logits[expert] = h.iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f32>();
        }
        let mut ranked = (0..n_expert).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let selected = &ranked[..n_expert_used];
        let max_selected = selected
            .iter()
            .map(|&expert| logits[expert])
            .fold(f32::NEG_INFINITY, f32::max);
        let selected_sum = selected
            .iter()
            .map(|&expert| (logits[expert] - max_selected).exp())
            .sum::<f32>();
        let gate_refs = selected
            .iter()
            .map(|&expert| gate[expert].as_slice())
            .collect::<Vec<_>>();
        let up_refs = selected
            .iter()
            .map(|&expert| up[expert].as_slice())
            .collect::<Vec<_>>();
        let down_refs = selected
            .iter()
            .map(|&expert| down[expert].as_slice())
            .collect::<Vec<_>>();
        let routes = selected
            .iter()
            .map(|&expert| (logits[expert] - max_selected).exp() / selected_sum)
            .collect::<Vec<_>>();
        let moe = cpu_qwen35_sparse_q4k_reference(
            &gate_refs, &up_refs, &down_refs, &routes, n_ff, hidden_dim, h,
        );
        for row in 0..hidden_dim {
            expected[token * hidden_dim + row] += moe[row];
        }
    }

    assert_close_rows_abs_rel(
        "MTP verify sparse MoE from device router",
        &actual,
        &expected,
        0.2,
        0.02,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_shared_expert_adds_device_route_residual() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4_f32_limit = usize::MAX;
    let window_tokens = 2usize;
    let hidden_dim = 256usize;
    let n_ff = 256usize;
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let verify_tokens = [13_u32, 14];
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify buffers");
    let hidden_after_sparse = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.0078125)
        .collect::<Vec<_>>();
    let moe_input = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.013671875)
        .collect::<Vec<_>>();
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_after_sparse.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_after_sparse.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                buffers.scratch_hidden_dev,
                moe_input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(moe_input.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let shared_input_scale = (0..hidden_dim)
        .map(|i| ((i as f32 % 7.0) - 3.0) * 0.015625)
        .collect::<Vec<_>>();
    let shared_gate = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 601)
        .into_iter()
        .next()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 607)
        .into_iter()
        .next()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, hidden_dim, n_ff / 256, 613)
        .into_iter()
        .next()
        .unwrap();

    state
        .stage_mtp_verify_gdn_shared_expert_q4k(
            &buffers,
            &shared_input_scale,
            &shared_gate,
            &shared_up,
            &shared_down,
            12,
            n_ff,
            hidden_dim,
        )
        .expect("stage shared expert");

    let mut actual = vec![0.0f32; hidden_after_sparse.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected = hidden_after_sparse.clone();
    for token in 0..window_tokens {
        let token_input = &moe_input[token * hidden_dim..(token + 1) * hidden_dim];
        let gate_dot = token_input
            .iter()
            .zip(shared_input_scale.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let route = 1.0 / (1.0 + (-gate_dot).exp());
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, hidden_dim / 256, token_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, hidden_dim / 256, token_input);
        let shared_hidden = shared_gate_out
            .iter()
            .zip(shared_up_out.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect::<Vec<_>>();
        let shared = cpu_q4k_gemv_rows(&shared_down, hidden_dim, n_ff / 256, &shared_hidden);
        for row in 0..hidden_dim {
            expected[token * hidden_dim + row] += shared[row] * route;
        }
    }

    assert_close_rows_abs_rel(
        "MTP verify shared expert residual",
        &actual,
        &expected,
        0.2,
        0.02,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_moe_residual_chains_router_sparse_and_shared() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4_f32_limit = usize::MAX;
    let window_tokens = 2usize;
    let hidden_dim = 256usize;
    let n_ff = 256usize;
    let n_expert = 3usize;
    let n_expert_used = 2usize;
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let verify_tokens = [17_u32, 18];
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify buffers");
    let hidden_before_moe = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 31.0) - 15.0) * 0.009765625)
        .collect::<Vec<_>>();
    let mut moe_input = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.01171875)
        .collect::<Vec<_>>();
    moe_input[0] = 1.25;
    moe_input[1] = 0.5;
    moe_input[hidden_dim] = 0.25;
    moe_input[hidden_dim + 1] = 1.5;
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_before_moe.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_before_moe.as_slice()),
                state.stream,
            )
            .unwrap();
        state
            .api
            .memcpy_htod_async(
                buffers.scratch_hidden_dev,
                moe_input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(moe_input.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let mut router = vec![0.0_f32; n_expert * hidden_dim];
    router[0] = 1.0;
    router[hidden_dim + 1] = 1.0;
    router[2 * hidden_dim] = 0.25;
    router[2 * hidden_dim + 1] = 0.25;
    let gate = make_test_q4k_weights(n_expert, n_ff, hidden_dim / 256, 631);
    let up = make_test_q4k_weights(n_expert, n_ff, hidden_dim / 256, 641);
    let down = make_test_q4k_weights(n_expert, hidden_dim, n_ff / 256, 643);
    let gate_all = gate.concat();
    let up_all = up.concat();
    let down_all = down.concat();
    let shared_input_scale = (0..hidden_dim)
        .map(|i| ((i as f32 % 11.0) - 5.0) * 0.009765625)
        .collect::<Vec<_>>();
    let shared_gate = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 647)
        .into_iter()
        .next()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 653)
        .into_iter()
        .next()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, hidden_dim, n_ff / 256, 659)
        .into_iter()
        .next()
        .unwrap();

    state
        .stage_mtp_verify_qwen35_moe_residual_q4k(
            &buffers,
            None,
            &router,
            n_expert,
            n_expert_used,
            &gate_all,
            &up_all,
            &down_all,
            12,
            &shared_input_scale,
            &shared_gate,
            &shared_up,
            &shared_down,
            12,
            n_ff,
            hidden_dim,
        )
        .expect("stage chained MoE residual");
    assert!(
        state.resident_moe_layers.is_empty(),
        "MTP verify MoE residual should use selected expert slots, not full-layer resident cache"
    );
    assert_eq!(state.resident_moe_layer_bytes, 0);

    let mut actual = vec![0.0f32; hidden_before_moe.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut expected = hidden_before_moe.clone();
    for token in 0..window_tokens {
        let token_input = &moe_input[token * hidden_dim..(token + 1) * hidden_dim];
        let mut logits = vec![0.0_f32; n_expert];
        for expert in 0..n_expert {
            let w = &router[expert * hidden_dim..(expert + 1) * hidden_dim];
            logits[expert] = token_input
                .iter()
                .zip(w.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
        }
        let mut ranked = (0..n_expert).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let selected = &ranked[..n_expert_used];
        let max_selected = selected
            .iter()
            .map(|&expert| logits[expert])
            .fold(f32::NEG_INFINITY, f32::max);
        let selected_sum = selected
            .iter()
            .map(|&expert| (logits[expert] - max_selected).exp())
            .sum::<f32>();
        let gate_refs = selected
            .iter()
            .map(|&expert| gate[expert].as_slice())
            .collect::<Vec<_>>();
        let up_refs = selected
            .iter()
            .map(|&expert| up[expert].as_slice())
            .collect::<Vec<_>>();
        let down_refs = selected
            .iter()
            .map(|&expert| down[expert].as_slice())
            .collect::<Vec<_>>();
        let routes = selected
            .iter()
            .map(|&expert| (logits[expert] - max_selected).exp() / selected_sum)
            .collect::<Vec<_>>();
        let sparse = cpu_qwen35_sparse_q4k_reference(
            &gate_refs,
            &up_refs,
            &down_refs,
            &routes,
            n_ff,
            hidden_dim,
            token_input,
        );
        let shared_gate_dot = token_input
            .iter()
            .zip(shared_input_scale.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let shared_route = 1.0 / (1.0 + (-shared_gate_dot).exp());
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, hidden_dim / 256, token_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, hidden_dim / 256, token_input);
        let shared_hidden = shared_gate_out
            .iter()
            .zip(shared_up_out.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect::<Vec<_>>();
        let shared = cpu_q4k_gemv_rows(&shared_down, hidden_dim, n_ff / 256, &shared_hidden);
        for row in 0..hidden_dim {
            expected[token * hidden_dim + row] += sparse[row] + shared[row] * shared_route;
        }
    }

    assert_close_rows_abs_rel(
        "MTP verify chained MoE residual",
        &actual,
        &expected,
        0.25,
        0.025,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_resident_moe_layer_opt_in_registers_first_layer() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let _resident_guard = EnvVarGuard::set("RNB_CUDA_MTP_VERIFY_RESIDENT_MOE_LAYER", "1");
    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_moe_layer_limit = usize::MAX;
    let window_tokens = 1usize;
    let hidden_dim = 256usize;
    let n_ff = 256usize;
    let n_expert = 2usize;
    let n_expert_used = 1usize;
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &[17_u32], &[])
        .expect("stage verify buffers");
    let hidden_before = (0..hidden_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.015625)
        .collect::<Vec<_>>();
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_before.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_before.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state
        .stage_mtp_verify_hidden_rows_rms_norm(&buffers, &vec![1.0; hidden_dim], 1.0e-5, false)
        .expect("stage MoE input norm");
    let mut router = vec![0.0f32; n_expert * hidden_dim];
    router[1] = 0.25;
    let gate = make_test_q4k_weights(n_expert, n_ff, hidden_dim / 256, 1701);
    let up = make_test_q4k_weights(n_expert, n_ff, hidden_dim / 256, 1709);
    let down = make_test_q4k_weights(n_expert, hidden_dim, n_ff / 256, 1717);
    let gate_all = gate.concat();
    let up_all = up.concat();
    let down_all = down.concat();
    let shared_input_scale = vec![0.0f32; hidden_dim];
    let shared_gate = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 1723)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 1733)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, hidden_dim, n_ff / 256, 1741)
        .pop()
        .unwrap();

    state
        .stage_mtp_verify_qwen35_moe_residual_q4k(
            &buffers,
            None,
            &router,
            n_expert,
            n_expert_used,
            &gate_all,
            &up_all,
            &down_all,
            GGML_Q4_K,
            &shared_input_scale,
            &shared_gate,
            &shared_up,
            &shared_down,
            GGML_Q4_K,
            n_ff,
            hidden_dim,
        )
        .expect("stage resident MoE residual");

    assert!(
        !state.resident_moe_layers.is_empty(),
        "resident MTP MoE opt-in should register a missing full layer before dispatch"
    );
    assert!(state.resident_moe_layer_bytes > 0);
}

#[test]
fn cuda_qwen35_mtp_verify_gdn_moe_layer_chains_resident_stages() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let mut state = CudaState::open().expect("open CUDA state");
    state.resident_q4_f32_limit = usize::MAX;
    let window_tokens = 2usize;
    let hidden_dim = 256usize;
    let num_k_heads = 1usize;
    let num_v_heads = 2usize;
    let head_k_dim = 4usize;
    let head_v_dim = 128usize;
    let d_inner = num_v_heads * head_v_dim;
    let conv_channels = num_k_heads * head_k_dim * 2 + d_inner;
    let kernel_size = 2usize;
    let n_ff = 256usize;
    let n_expert = 3usize;
    let n_expert_used = 2usize;
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let verify_tokens = [21_u32, 22];
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify buffers");
    let hidden_before = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.0078125)
        .collect::<Vec<_>>();
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_before.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_before.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    let attn_norm = (0..hidden_dim)
        .map(|i| 0.75 + (i % 13) as f32 * 0.0078125)
        .collect::<Vec<_>>();
    let qkv = make_test_q4k_weights(1, conv_channels, hidden_dim / 256, 701)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, d_inner, hidden_dim / 256, 709)
        .pop()
        .unwrap();
    let alpha = make_test_q4k_weights(1, num_v_heads, hidden_dim / 256, 719)
        .pop()
        .unwrap();
    let beta = make_test_q4k_weights(1, num_v_heads, hidden_dim / 256, 727)
        .pop()
        .unwrap();
    let conv_state = (0..(kernel_size - 1) * conv_channels)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.01171875)
        .collect::<Vec<_>>();
    let conv_kernel = (0..kernel_size * conv_channels)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.009765625)
        .collect::<Vec<_>>();
    let dt_bias = [-0.25_f32, 0.125];
    let ssm_a = [-0.75_f32, -0.5];
    let mut delta_state = (0..num_v_heads * head_v_dim * head_k_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.005859375)
        .collect::<Vec<_>>();
    let mut expected_delta_state = delta_state.clone();
    let ssm_norm = (0..head_v_dim)
        .map(|i| 0.5 + (i % 17) as f32 * 0.00390625)
        .collect::<Vec<_>>();
    let ssm_out = make_test_q4k_weights(1, hidden_dim, d_inner / 256, 733)
        .pop()
        .unwrap();
    let post_attn_norm = (0..hidden_dim)
        .map(|i| 0.625 + (i % 11) as f32 * 0.005859375)
        .collect::<Vec<_>>();
    let mut router = vec![0.0_f32; n_expert * hidden_dim];
    router[0] = 1.0;
    router[hidden_dim + 1] = 1.0;
    router[2 * hidden_dim] = 0.25;
    router[2 * hidden_dim + 1] = 0.5;
    let expert_gate = make_test_q4k_weights(n_expert, n_ff, hidden_dim / 256, 739);
    let expert_up = make_test_q4k_weights(n_expert, n_ff, hidden_dim / 256, 743);
    let expert_down = make_test_q4k_weights(n_expert, hidden_dim, n_ff / 256, 751);
    let gate_all = expert_gate.concat();
    let up_all = expert_up.concat();
    let down_all = expert_down.concat();
    let shared_input_scale = (0..hidden_dim)
        .map(|i| ((i as f32 % 7.0) - 3.0) * 0.0078125)
        .collect::<Vec<_>>();
    let shared_gate = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 757)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 761)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, hidden_dim, n_ff / 256, 769)
        .pop()
        .unwrap();

    let prefix_tokens = [1usize];
    let state_capture = state
        .stage_mtp_verify_qwen35_gdn_moe_layer_q4k_capture_states(
            &buffers,
            7,
            mtp_verify::Qwen35MtpGdnMoeLayerRequest {
                projection: mtp_verify::Qwen35MtpGdnProjectionRequest {
                    attn_norm: &attn_norm,
                    qkv_q4k: &qkv,
                    qkv_quant: 12,
                    qkv_rows: conv_channels,
                    qkv_cols: hidden_dim,
                    gate_q4k: &gate,
                    gate_rows: d_inner,
                    gate_cols: hidden_dim,
                    alpha_q4k: &alpha,
                    alpha_f32: &[],
                    alpha_quant: GGML_Q4_K,
                    alpha_rows: num_v_heads,
                    alpha_cols: hidden_dim,
                    beta_q4k: &beta,
                    beta_f32: &[],
                    beta_quant: GGML_Q4_K,
                    beta_rows: num_v_heads,
                    beta_cols: hidden_dim,
                    norm_eps: 1.0e-5,
                },
                conv_state: &conv_state,
                conv_kernel: &conv_kernel,
                kernel_size,
                dt_bias: &dt_bias,
                ssm_a: &ssm_a,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                delta_state: &mut delta_state,
                sync_delta_state_to_host: true,
                ssm_norm: &ssm_norm,
                ssm_out_q4k: &ssm_out,
                ssm_out_quant: GGML_Q4_K,
                ssm_out_rows: hidden_dim,
                ssm_out_cols: d_inner,
                post_attn_norm: &post_attn_norm,
                router_w: &router,
                n_expert,
                n_expert_used,
                gate_all: &gate_all,
                up_all: &up_all,
                down_all: &down_all,
                down_quant: 12,
                shared_input_scale: &shared_input_scale,
                shared_gate: &shared_gate,
                shared_up: &shared_up,
                shared_down: &shared_down,
                shared_down_quant: 12,
                n_ff,
                n_embd: hidden_dim,
                ffn_gate_q4k: &[],
                ffn_gate_rows: 0,
                ffn_gate_cols: 0,
                ffn_up_q4k: &[],
                ffn_up_rows: 0,
                ffn_up_cols: 0,
                ffn_down: &[],
                ffn_down_quant: 12,
                ffn_down_rows: 0,
                ffn_down_cols: 0,
                norm_eps: 1.0e-5,
            },
            &prefix_tokens,
        )
        .expect("stage GDN MoE layer");
    let prefix_states = &state_capture.prefix_states;

    let mut actual = vec![0.0f32; hidden_before.len()];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.hidden_rows_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let mut qkv_rows = Vec::with_capacity(window_tokens * conv_channels);
    let mut gate_rows = Vec::with_capacity(window_tokens * d_inner);
    let mut alpha_rows = Vec::with_capacity(window_tokens * num_v_heads);
    let mut beta_rows = Vec::with_capacity(window_tokens * num_v_heads);
    for hidden in hidden_before.chunks_exact(hidden_dim) {
        let normed = cpu_rms_norm(hidden, &attn_norm, 1.0e-5, false);
        qkv_rows.extend(cpu_q4k_gemv_rows(
            &qkv,
            conv_channels,
            hidden_dim / 256,
            &normed,
        ));
        gate_rows.extend(cpu_q4k_gemv_rows(&gate, d_inner, hidden_dim / 256, &normed));
        alpha_rows.extend(cpu_q4k_gemv_rows(
            &alpha,
            num_v_heads,
            hidden_dim / 256,
            &normed,
        ));
        beta_rows.extend(cpu_q4k_gemv_rows(
            &beta,
            num_v_heads,
            hidden_dim / 256,
            &normed,
        ));
    }
    let mut conv_input = conv_state.clone();
    conv_input.extend_from_slice(&qkv_rows);
    let mut conv_out = vec![0.0f32; window_tokens * conv_channels];
    for token_idx in 0..window_tokens {
        for channel_idx in 0..conv_channels {
            let mut sum = 0.0f32;
            for k in 0..kernel_size {
                sum += conv_input[(token_idx + k) * conv_channels + channel_idx]
                    * conv_kernel[k * conv_channels + channel_idx];
            }
            conv_out[token_idx * conv_channels + channel_idx] = sum / (1.0 + (-sum).exp());
        }
    }
    let q_dim = num_k_heads * head_k_dim;
    let k_dim = q_dim;
    let q_scale = 1.0 / (head_k_dim as f32).sqrt();
    let mut q = vec![0.0f32; window_tokens * num_v_heads * head_k_dim];
    let mut k = vec![0.0f32; q.len()];
    let mut v = vec![0.0f32; window_tokens * num_v_heads * head_v_dim];
    let mut delta_gate = vec![0.0f32; window_tokens * num_v_heads];
    let mut delta_beta = vec![0.0f32; window_tokens * num_v_heads];
    for token_idx in 0..window_tokens {
        let row = &conv_out[token_idx * conv_channels..(token_idx + 1) * conv_channels];
        for k_head in 0..num_k_heads {
            let q_src = &row[k_head * head_k_dim..(k_head + 1) * head_k_dim];
            let k_src = &row[q_dim + k_head * head_k_dim..q_dim + (k_head + 1) * head_k_dim];
            let q_inv =
                1.0 / (q_src.iter().map(|value| value * value).sum::<f32>() + 1.0e-5).sqrt();
            let k_inv =
                1.0 / (k_src.iter().map(|value| value * value).sum::<f32>() + 1.0e-5).sqrt();
            for v_head in (k_head..num_v_heads).step_by(num_k_heads) {
                let out = (token_idx * num_v_heads + v_head) * head_k_dim;
                for dim in 0..head_k_dim {
                    q[out + dim] = q_src[dim] * q_inv * q_scale;
                    k[out + dim] = k_src[dim] * k_inv;
                }
            }
        }
        for v_head in 0..num_v_heads {
            let v_src = q_dim + k_dim + v_head * head_v_dim;
            let v_out = (token_idx * num_v_heads + v_head) * head_v_dim;
            v[v_out..v_out + head_v_dim].copy_from_slice(&row[v_src..v_src + head_v_dim]);
            let gate_idx = token_idx * num_v_heads + v_head;
            let biased = alpha_rows[gate_idx] + dt_bias[v_head];
            delta_gate[gate_idx] = (1.0 + biased.exp()).ln() * ssm_a[v_head];
            delta_beta[gate_idx] = 1.0 / (1.0 + (-beta_rows[gate_idx]).exp());
        }
    }
    let delta_out = cpu_delta_net_prefill_reference(
        &mut expected_delta_state,
        &q,
        &k,
        &v,
        &delta_gate,
        &delta_beta,
        window_tokens,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    );
    let mut expected_prefix_delta_state = (0..num_v_heads * head_v_dim * head_k_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.005859375)
        .collect::<Vec<_>>();
    cpu_delta_net_prefill_reference(
        &mut expected_prefix_delta_state,
        &q[..num_v_heads * head_k_dim],
        &k[..num_v_heads * head_k_dim],
        &v[..num_v_heads * head_v_dim],
        &delta_gate[..num_v_heads],
        &delta_beta[..num_v_heads],
        1,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    );
    let mut gated = vec![0.0f32; window_tokens * d_inner];
    for row_idx in 0..window_tokens * num_v_heads {
        let start = row_idx * head_v_dim;
        let row = &delta_out[start..start + head_v_dim];
        let inv = 1.0
            / (row.iter().map(|value| value * value).sum::<f32>() / head_v_dim as f32 + 1.0e-5)
                .sqrt();
        for dim in 0..head_v_dim {
            let idx = start + dim;
            let z = gate_rows[idx];
            gated[idx] = row[dim] * inv * ssm_norm[dim] * (z / (1.0 + (-z).exp()));
        }
    }
    let mut expected = hidden_before.clone();
    for token_idx in 0..window_tokens {
        let input = &gated[token_idx * d_inner..(token_idx + 1) * d_inner];
        let ssm = cpu_q4k_gemv_rows(&ssm_out, hidden_dim, d_inner / 256, input);
        for row in 0..hidden_dim {
            expected[token_idx * hidden_dim + row] += ssm[row];
        }
    }
    let mut scratch = Vec::with_capacity(expected.len());
    for row in expected.chunks_exact(hidden_dim) {
        scratch.extend(cpu_rms_norm(row, &post_attn_norm, 1.0e-5, false));
    }
    for token in 0..window_tokens {
        let token_input = &scratch[token * hidden_dim..(token + 1) * hidden_dim];
        let mut logits = vec![0.0_f32; n_expert];
        for expert in 0..n_expert {
            let w = &router[expert * hidden_dim..(expert + 1) * hidden_dim];
            logits[expert] = token_input
                .iter()
                .zip(w.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
        }
        let mut ranked = (0..n_expert).collect::<Vec<_>>();
        ranked.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let selected = &ranked[..n_expert_used];
        let max_selected = selected
            .iter()
            .map(|&expert| logits[expert])
            .fold(f32::NEG_INFINITY, f32::max);
        let selected_sum = selected
            .iter()
            .map(|&expert| (logits[expert] - max_selected).exp())
            .sum::<f32>();
        let gate_refs = selected
            .iter()
            .map(|&expert| expert_gate[expert].as_slice())
            .collect::<Vec<_>>();
        let up_refs = selected
            .iter()
            .map(|&expert| expert_up[expert].as_slice())
            .collect::<Vec<_>>();
        let down_refs = selected
            .iter()
            .map(|&expert| expert_down[expert].as_slice())
            .collect::<Vec<_>>();
        let routes = selected
            .iter()
            .map(|&expert| (logits[expert] - max_selected).exp() / selected_sum)
            .collect::<Vec<_>>();
        let sparse = cpu_qwen35_sparse_q4k_reference(
            &gate_refs,
            &up_refs,
            &down_refs,
            &routes,
            n_ff,
            hidden_dim,
            token_input,
        );
        let shared_gate_dot = token_input
            .iter()
            .zip(shared_input_scale.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let shared_route = 1.0 / (1.0 + (-shared_gate_dot).exp());
        let shared_gate_out = cpu_q4k_gemv_rows(&shared_gate, n_ff, hidden_dim / 256, token_input);
        let shared_up_out = cpu_q4k_gemv_rows(&shared_up, n_ff, hidden_dim / 256, token_input);
        let shared_hidden = shared_gate_out
            .iter()
            .zip(shared_up_out.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect::<Vec<_>>();
        let shared = cpu_q4k_gemv_rows(&shared_down, hidden_dim, n_ff / 256, &shared_hidden);
        for row in 0..hidden_dim {
            expected[token * hidden_dim + row] += sparse[row] + shared[row] * shared_route;
        }
    }

    assert_close_rows_abs_rel(
        "MTP verify GDN MoE layer hidden",
        &actual,
        &expected,
        0.35,
        0.03,
    );
    assert_close_rows(
        "MTP verify GDN MoE layer delta state",
        &delta_state,
        &expected_delta_state,
        2.0e-5,
    );
    assert_eq!(prefix_states.len(), 1);
    assert_eq!(prefix_states[0].prefix_tokens, 1);
    assert_eq!(prefix_states[0].layers.len(), 1);
    assert_eq!(prefix_states[0].layers[0].layer_idx, 7);
    assert_close_rows(
        "MTP verify GDN MoE layer prefix conv state",
        &prefix_states[0].layers[0].conv_state,
        &conv_input[conv_channels..2 * conv_channels],
        1.0e-4,
    );
    let snapshot = prefix_states[0].layers[0]
        .resident_delta_snapshot
        .as_ref()
        .expect("resident prefix delta snapshot");
    assert!(state
        .restore_resident_delta_state(&mut delta_state, snapshot)
        .expect("restore prefix resident delta state"));
    assert!(state
        .sync_resident_delta_state(&mut delta_state)
        .expect("sync prefix resident delta state"));
    assert_close_rows(
        "MTP verify GDN MoE layer prefix delta state",
        &delta_state,
        &expected_prefix_delta_state,
        2.0e-5,
    );
    assert_eq!(state_capture.final_state.layer_idx, 7);
    assert_close_rows(
        "MTP verify GDN MoE layer final conv state",
        &state_capture.final_state.conv_state,
        &conv_input[window_tokens * conv_channels..(window_tokens + 1) * conv_channels],
        1.0e-4,
    );
}

#[test]
fn cuda_qwen35_mtp_verify_output_argmax_writes_target_tokens() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 3usize;
    let vocab_rows = 19usize;
    let output_weight = make_test_q6k_weights(1, vocab_rows, hidden_dim / 256, 313)
        .pop()
        .unwrap();
    let output_norm = (0..hidden_dim)
        .map(|i| 0.75 + ((i % 11) as f32) * 0.03125)
        .collect::<Vec<_>>();
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.017578125)
        .collect::<Vec<_>>();
    let verify_tokens = [4_u32, 5, 6];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    state
        .stage_mtp_verify_output_argmax_q6k(
            &buffers,
            &output_weight,
            vocab_rows,
            hidden_dim,
            &output_norm,
            1.0e-5,
        )
        .expect("write target token argmax");

    let mut actual = vec![0_u32; window_tokens];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.target_tokens_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let expected = hidden_rows
        .chunks_exact(hidden_dim)
        .map(|hidden| {
            let normed = cpu_rms_norm(hidden, &output_norm, 1.0e-5, false);
            let logits = cpu_q6k_gemv_rows(&output_weight, vocab_rows, hidden_dim / 256, &normed);
            logits
                .iter()
                .enumerate()
                .max_by(|(a_idx, a), (b_idx, b)| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b_idx.cmp(a_idx))
                })
                .map(|(idx, _)| idx as u32)
                .unwrap()
        })
        .collect::<Vec<_>>();

    assert_eq!(actual, expected);
}

#[test]
fn cuda_qwen35_mtp_verify_q6k_output_argmax_batched_matches_per_token_cpu() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 3usize;
    let vocab_rows = 23usize;
    let output_weight = make_test_q6k_weights(1, vocab_rows, hidden_dim / 256, 719)
        .pop()
        .unwrap();
    let normed_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 37.0) - 18.0) * 0.013671875)
        .collect::<Vec<_>>();
    let verify_tokens = [11_u32, 12, 13];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.scratch_hidden_dev,
                normed_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(normed_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    state
        .write_q6k_argmax_tokens_batched_from_dev_input(
            &output_weight,
            vocab_rows,
            hidden_dim / 256,
            buffers.scratch_hidden_dev,
            hidden_dim,
            window_tokens,
            buffers.target_tokens_dev,
        )
        .expect("write batched q6k argmax tokens");

    let mut actual = vec![0_u32; window_tokens];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.target_tokens_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let expected = normed_rows
        .chunks_exact(hidden_dim)
        .map(|row| {
            let logits = cpu_q6k_gemv_rows(&output_weight, vocab_rows, hidden_dim / 256, row);
            logits
                .iter()
                .enumerate()
                .max_by(|(a_idx, a), (b_idx, b)| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b_idx.cmp(a_idx))
                })
                .map(|(idx, _)| idx as u32)
                .unwrap()
        })
        .collect::<Vec<_>>();

    assert_eq!(actual, expected);
}

#[test]
fn cuda_q4k_pinned_resident_survives_lru_pressure() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let pinned = vec![1_u8; 4096];
    let pressure_a = vec![2_u8; 4096];
    let pressure_b = vec![3_u8; 4096];
    state.resident_q4k_limit = pinned.len() + pressure_a.len();

    state
        .resident_q4k_weights_ptr_pinned(&pinned)
        .expect("pin resident q4k weight");
    state
        .resident_q4k_weights_ptr(&pressure_a)
        .expect("stage first pressure weight");
    state
        .resident_q4k_weights_ptr(&pressure_b)
        .expect("stage second pressure weight");

    assert!(
        state.resident_q4k.contains_key(&q4k_resident_key(&pinned)),
        "pinned q4k resident entry must survive LRU pressure"
    );
}

#[test]
fn cuda_q4k_arena_bytes_include_pinned_resident_entries() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let pinned = vec![1_u8; 4096];
    let arena_weight = vec![2_u8; 4096];
    state.resident_q4k_limit = pinned.len() + arena_weight.len();

    state
        .resident_q4k_weights_ptr_pinned(&pinned)
        .expect("pin resident q4k weight");
    state
        .resident_q4k_weights_ptr_current_arena(&arena_weight)
        .expect("stage arena resident q4k weight");

    assert!(
        state.resident_q4k_bytes >= pinned.len() + arena_weight.len(),
        "arena resident byte accounting must include pinned entries"
    );
}

#[test]
fn cuda_q4k_offload_preserves_pinned_resident_entries() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let pinned = vec![1_u8; 4096];
    let arena_weight = vec![2_u8; 4096];
    let owned_weight = vec![3_u8; 4096];
    state.resident_q4k_limit = pinned.len() + arena_weight.len() + owned_weight.len();

    state
        .resident_q4k_weights_ptr_pinned(&pinned)
        .expect("pin resident q4k weight");
    state
        .resident_q4k_weights_ptr_current_arena(&arena_weight)
        .expect("stage arena resident q4k weight");
    state
        .resident_q4k_weights_ptr(&owned_weight)
        .expect("stage owned resident q4k weight");

    let released = state
        .offload_non_pinned_resident_q4k()
        .expect("offload non-pinned q4k residents");

    assert!(released >= arena_weight.len() + owned_weight.len());
    assert!(state.resident_q4k.contains_key(&q4k_resident_key(&pinned)));
    assert!(!state
        .resident_q4k
        .contains_key(&q4k_resident_key(&arena_weight)));
    assert!(!state
        .resident_q4k
        .contains_key(&q4k_resident_key(&owned_weight)));
    assert_eq!(state.resident_q4k_bytes, pinned.len());
}

#[test]
fn cuda_f16_gemv_matches_cpu_reference() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let rows = 3usize;
    let cols = 4usize;
    let weights_f32 = [
        0.5_f32, -1.0, 0.25, 2.0, -0.75, 0.125, 1.5, -2.0, 1.0, 0.0, -0.5, 0.75,
    ];
    let mut weights = Vec::with_capacity(weights_f32.len() * 2);
    for value in weights_f32 {
        weights.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
    }
    let input = [1.0_f32, -2.0, 0.5, 3.0];

    let actual = state
        .f16_gemv(&weights, rows, cols, &input)
        .expect("f16 gemv");

    let expected = weights_f32
        .chunks_exact(cols)
        .map(|row| {
            row.iter()
                .zip(input.iter())
                .map(|(&w, &x)| half::f16::from_f32(w).to_f32() * x)
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    assert_close_rows("CUDA F16 GEMV", &actual, &expected, 1.0e-4);
}

#[test]
fn cuda_qwen35_mtp_verify_output_argmax_q8_0_writes_target_tokens() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 3usize;
    let vocab_rows = 19usize;
    let output_weight = make_test_q8_0_weights(vocab_rows, hidden_dim, 313);
    let output_norm = (0..hidden_dim)
        .map(|i| 0.75 + ((i % 11) as f32) * 0.03125)
        .collect::<Vec<_>>();
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.017578125)
        .collect::<Vec<_>>();
    let verify_tokens = [4_u32, 5, 6];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    state
        .stage_mtp_verify_output_argmax_q8_0(
            &buffers,
            &output_weight,
            vocab_rows,
            hidden_dim,
            &output_norm,
            1.0e-5,
        )
        .expect("write Q8_0 target token argmax");

    let mut actual = vec![0_u32; window_tokens];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.target_tokens_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let expected = hidden_rows
        .chunks_exact(hidden_dim)
        .map(|hidden| {
            let normed = cpu_rms_norm(hidden, &output_norm, 1.0e-5, false);
            let logits = cpu_q8_0_rows(&output_weight, vocab_rows, hidden_dim, &normed);
            logits
                .iter()
                .enumerate()
                .max_by(|(a_idx, a), (b_idx, b)| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b_idx.cmp(a_idx))
                })
                .map(|(idx, _)| idx as u32)
                .unwrap()
        })
        .collect::<Vec<_>>();

    assert_eq!(actual, expected);
}

#[test]
fn cuda_qwen35_mtp_verify_output_argmax_q4k_writes_target_tokens() {
    let _guard = runtime_test_lock();
    let mut state = CudaState::open().expect("open CUDA state");
    let hidden_dim = 512usize;
    let window_tokens = 3usize;
    let vocab_rows = 19usize;
    let output_weight = make_test_q4k_weights(1, vocab_rows, hidden_dim / 256, 313)
        .pop()
        .unwrap();
    let output_norm = (0..hidden_dim)
        .map(|i| 0.75 + ((i % 11) as f32) * 0.03125)
        .collect::<Vec<_>>();
    let hidden_rows = (0..window_tokens * hidden_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.017578125)
        .collect::<Vec<_>>();
    let verify_tokens = [4_u32, 5, 6];
    let plan = qwen35_mtp_verify_buffer_plan(window_tokens, hidden_dim, 0).unwrap();
    let buffers = state
        .stage_mtp_verify_window(&plan, &verify_tokens, &[])
        .expect("stage verify tokens");
    unsafe {
        state
            .api
            .memcpy_htod_async(
                buffers.hidden_rows_dev,
                hidden_rows.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(hidden_rows.as_slice()),
                state.stream,
            )
            .unwrap();
    }

    state
        .stage_mtp_verify_output_argmax_q4k(
            &buffers,
            &output_weight,
            vocab_rows,
            hidden_dim,
            &output_norm,
            1.0e-5,
        )
        .expect("write Q4_K target token argmax");

    let mut actual = vec![0_u32; window_tokens];
    unsafe {
        state
            .api
            .memcpy_dtoh_async(
                actual.as_mut_ptr().cast::<libc::c_void>(),
                buffers.target_tokens_dev,
                std::mem::size_of_val(actual.as_slice()),
                state.stream,
            )
            .unwrap();
    }
    state.stream_synchronize().unwrap();

    let expected = hidden_rows
        .chunks_exact(hidden_dim)
        .map(|hidden| {
            let normed = cpu_rms_norm(hidden, &output_norm, 1.0e-5, false);
            let logits = cpu_q4k_gemv_rows(&output_weight, vocab_rows, hidden_dim / 256, &normed);
            logits
                .iter()
                .enumerate()
                .max_by(|(a_idx, a), (b_idx, b)| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b_idx.cmp(a_idx))
                })
                .map(|(idx, _)| idx as u32)
                .unwrap()
        })
        .collect::<Vec<_>>();

    assert_eq!(actual, expected);
}

#[test]
fn cuda_qwen35_mtp_device_verify_api_stages_before_unimplemented() {
    let _guard = runtime_test_lock();
    let verify_tokens = [10_u32, 11];
    let prefix_tokens = [1_usize];
    let hidden_dim = 512usize;
    let token_embd_rows = 16usize;
    let token_embd = make_test_q4k_weights(1, token_embd_rows, hidden_dim / 256, 197)
        .pop()
        .unwrap();
    let output_rows = 19usize;
    let output_q6k = make_test_q6k_weights(1, output_rows, hidden_dim / 256, 211)
        .pop()
        .unwrap();
    let output_norm = vec![1.0f32; hidden_dim];
    let attention_moe_layers = [];
    let mut gdn_moe_layers = [];
    let layer_order = [];
    let request = Qwen35MtpDeviceVerifyRequest {
        verify_tokens: &verify_tokens,
        prefix_tokens: &prefix_tokens,
        pos_start: 7,
        hidden_dim,
        rope_dim: 256,
        rope_neox: true,
        rope_theta: 10000.0,
        include_bonus: false,
        token_embd_q4k: &token_embd,
        token_embd_quant: GGML_Q4_K,
        token_embd_rows,
        token_embd_cols: hidden_dim,
        layer_order: &layer_order,
        attention_moe_layers: &attention_moe_layers,
        gdn_moe_layers: &mut gdn_moe_layers,
        output_q6k: &output_q6k,
        output_quant: GGML_Q6_K,
        output_rows,
        output_cols: hidden_dim,
        output_norm: &output_norm,
        norm_eps: 1.0e-5,
    };

    let err = match qwen35_mtp_device_verify_window(request) {
        Ok(_) => panic!("MTP device verify graph should still be unimplemented"),
        Err(err) => err,
    };

    assert!(err.contains("not implemented"));
    assert!(err.contains("staged=true"));
    assert!(err.contains("embeddings_staged=true"));
    assert!(err.contains("output_argmax_staged=true"));
    assert!(err.contains("gdn_moe_layers_staged=0"));
    assert!(err.contains("pos_start=7"));
}

#[test]
fn cuda_qwen35_mtp_device_verify_api_executes_attention_projection_before_unimplemented() {
    let _guard = runtime_test_lock();
    let verify_tokens = [10_u32, 11];
    let prefix_tokens = [1_usize];
    let hidden_dim = 512usize;
    let token_embd_rows = 16usize;
    let token_embd = make_test_q4k_weights(1, token_embd_rows, hidden_dim / 256, 197)
        .pop()
        .unwrap();
    let output_rows = 19usize;
    let output_q6k = make_test_q6k_weights(1, output_rows, hidden_dim / 256, 211)
        .pop()
        .unwrap();
    let output_norm = vec![1.0f32; hidden_dim];
    let attention_layer = Qwen35MtpDeviceVerifyAttentionMoeLayer {
        layer_index: 0,
        attn_norm: &[],
        q_q4k: &[],
        q_quant: 12,
        q_rows: 0,
        q_cols: 0,
        k_q4k: &[],
        k_quant: 12,
        k_rows: 0,
        k_cols: 0,
        v_q4k: &[],
        v_quant: 12,
        v_rows: 0,
        v_cols: 0,
        prior_k_bits: &[],
        prior_v_bits: &[],
        prior_tokens: 0,
        o_q4k: &[],
        o_quant: GGML_Q4_K,
        o_rows: 0,
        o_cols: 0,
        q_norm: &[],
        k_norm: &[],
        post_attn_norm: &[],
        ffn_norm: &[],
        ffn_gate_q4k: &[],
        ffn_gate_rows: 0,
        ffn_gate_cols: 0,
        ffn_up_q4k: &[],
        ffn_up_rows: 0,
        ffn_up_cols: 0,
        ffn_down: &[],
        ffn_down_quant: 12,
        ffn_down_rows: 0,
        ffn_down_cols: 0,
        router_w: &[],
        n_expert: 0,
        n_expert_used: 0,
        gate_all: &[],
        up_all: &[],
        down_all: &[],
        down_quant: 12,
        shared_input_scale: &[],
        shared_gate: &[],
        shared_up: &[],
        shared_down: &[],
        shared_down_quant: 12,
        n_ff: 0,
        n_embd: hidden_dim,
    };
    let attention_moe_layers = [attention_layer];
    let mut gdn_moe_layers = [];
    let layer_order = [];
    let request = Qwen35MtpDeviceVerifyRequest {
        verify_tokens: &verify_tokens,
        prefix_tokens: &prefix_tokens,
        pos_start: 7,
        hidden_dim,
        rope_dim: 256,
        rope_neox: true,
        rope_theta: 10000.0,
        include_bonus: false,
        token_embd_q4k: &token_embd,
        token_embd_quant: GGML_Q4_K,
        token_embd_rows,
        token_embd_cols: hidden_dim,
        layer_order: &layer_order,
        attention_moe_layers: &attention_moe_layers,
        gdn_moe_layers: &mut gdn_moe_layers,
        output_q6k: &output_q6k,
        output_quant: GGML_Q6_K,
        output_rows,
        output_cols: hidden_dim,
        output_norm: &output_norm,
        norm_eps: 1.0e-5,
    };

    let err = match qwen35_mtp_device_verify_window(request) {
        Ok(_) => panic!("invalid attention projection should not reach success"),
        Err(err) => err,
    };

    assert!(err.contains("attention attn_norm length mismatch"));
    assert!(!err.contains("attention layer graph is not implemented"));
}

#[test]
fn cuda_qwen35_mtp_device_verify_api_executes_attention_qk_norm_rope_before_unimplemented() {
    let _guard = runtime_test_lock();
    let verify_tokens = [10_u32, 11];
    let prefix_tokens = [1_usize];
    let hidden_dim = 512usize;
    let head_dim = 256usize;
    let token_embd_rows = 16usize;
    let token_embd = make_test_q4k_weights(1, token_embd_rows, hidden_dim / 256, 197)
        .pop()
        .unwrap();
    let q = make_test_q4k_weights(1, 2 * head_dim, hidden_dim / 256, 251)
        .pop()
        .unwrap();
    let k = make_test_q4k_weights(1, head_dim, hidden_dim / 256, 257)
        .pop()
        .unwrap();
    let v = make_test_q4k_weights(1, head_dim, hidden_dim / 256, 263)
        .pop()
        .unwrap();
    let output_rows = 19usize;
    let output_q6k = make_test_q6k_weights(1, output_rows, hidden_dim / 256, 211)
        .pop()
        .unwrap();
    let output_norm = vec![1.0f32; hidden_dim];
    let attn_norm = vec![1.0f32; hidden_dim];
    let k_norm = vec![1.0f32; head_dim];
    let attention_layer = Qwen35MtpDeviceVerifyAttentionMoeLayer {
        layer_index: 0,
        attn_norm: &attn_norm,
        q_q4k: &q,
        q_quant: 12,
        q_rows: 2 * head_dim,
        q_cols: hidden_dim,
        k_q4k: &k,
        k_quant: 12,
        k_rows: head_dim,
        k_cols: hidden_dim,
        v_q4k: &v,
        v_quant: 12,
        v_rows: head_dim,
        v_cols: hidden_dim,
        prior_k_bits: &[],
        prior_v_bits: &[],
        prior_tokens: 0,
        o_q4k: &[],
        o_quant: GGML_Q4_K,
        o_rows: 0,
        o_cols: 0,
        q_norm: &[],
        k_norm: &k_norm,
        post_attn_norm: &[],
        ffn_norm: &[],
        ffn_gate_q4k: &[],
        ffn_gate_rows: 0,
        ffn_gate_cols: 0,
        ffn_up_q4k: &[],
        ffn_up_rows: 0,
        ffn_up_cols: 0,
        ffn_down: &[],
        ffn_down_quant: 12,
        ffn_down_rows: 0,
        ffn_down_cols: 0,
        router_w: &[],
        n_expert: 0,
        n_expert_used: 0,
        gate_all: &[],
        up_all: &[],
        down_all: &[],
        down_quant: 12,
        shared_input_scale: &[],
        shared_gate: &[],
        shared_up: &[],
        shared_down: &[],
        shared_down_quant: 12,
        n_ff: 0,
        n_embd: hidden_dim,
    };
    let attention_moe_layers = [attention_layer];
    let mut gdn_moe_layers = [];
    let layer_order = [];
    let request = Qwen35MtpDeviceVerifyRequest {
        verify_tokens: &verify_tokens,
        prefix_tokens: &prefix_tokens,
        pos_start: 7,
        hidden_dim,
        rope_dim: 256,
        rope_neox: true,
        rope_theta: 10000.0,
        include_bonus: false,
        token_embd_q4k: &token_embd,
        token_embd_quant: GGML_Q4_K,
        token_embd_rows,
        token_embd_cols: hidden_dim,
        layer_order: &layer_order,
        attention_moe_layers: &attention_moe_layers,
        gdn_moe_layers: &mut gdn_moe_layers,
        output_q6k: &output_q6k,
        output_quant: GGML_Q6_K,
        output_rows,
        output_cols: hidden_dim,
        output_norm: &output_norm,
        norm_eps: 1.0e-5,
    };

    let err = match qwen35_mtp_device_verify_window(request) {
        Ok(_) => panic!("invalid q_norm should not reach success"),
        Err(err) => err,
    };

    assert!(err.contains("attention q_norm length mismatch"));
    assert!(!err.contains("attention layer graph is not implemented"));
}

#[test]
fn cuda_qwen35_mtp_device_verify_api_returns_attention_kv_states() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    allow_default_cuda_q4_f32_cache_for_test();
    let verify_tokens = [10_u32, 11];
    let prefix_tokens = [1_usize];
    let hidden_dim = 512usize;
    let head_dim = 256usize;
    let n_ff = 256usize;
    let n_expert = 3usize;
    let n_expert_used = 2usize;
    let blocks_per_row = hidden_dim / 256;
    let token_embd_rows = 16usize;
    let token_embd = make_test_q4k_weights(1, token_embd_rows, blocks_per_row, 197)
        .pop()
        .unwrap();
    let q = make_test_q4k_weights(1, 2 * head_dim, blocks_per_row, 271)
        .pop()
        .unwrap();
    let k = make_test_q4k_weights(1, head_dim, blocks_per_row, 277)
        .pop()
        .unwrap();
    let v = make_test_q6k_weights(1, head_dim, blocks_per_row, 281)
        .pop()
        .unwrap();
    let o = make_test_q4k_weights(1, hidden_dim, blocks_per_row, 283)
        .pop()
        .unwrap();
    let output_rows = 19usize;
    let output_q6k = make_test_q6k_weights(1, output_rows, blocks_per_row, 211)
        .pop()
        .unwrap();
    let output_norm = vec![1.0f32; hidden_dim];
    let attn_norm = vec![1.0f32; hidden_dim];
    let q_norm = vec![1.0f32; head_dim];
    let k_norm = vec![1.0f32; head_dim];
    let post_attn_norm = vec![1.0f32; hidden_dim];
    let ffn_norm = vec![1.0f32; hidden_dim];
    let mut router = vec![0.0_f32; n_expert * hidden_dim];
    router[0] = 1.0;
    router[hidden_dim + 1] = 1.0;
    router[2 * hidden_dim] = 0.25;
    router[2 * hidden_dim + 1] = 0.25;
    let gate_all = make_test_q4k_weights(n_expert, n_ff, blocks_per_row, 291).concat();
    let up_all = make_test_q4k_weights(n_expert, n_ff, blocks_per_row, 293).concat();
    let down_all = make_test_q4k_weights(n_expert, hidden_dim, n_ff / 256, 307).concat();
    let shared_input_scale = vec![0.0f32; hidden_dim];
    let shared_gate = make_test_q4k_weights(1, n_ff, blocks_per_row, 311)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, blocks_per_row, 313)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, hidden_dim, n_ff / 256, 317)
        .pop()
        .unwrap();
    let prior_tokens = 2usize;
    let prior_k_bits = (0..prior_tokens * head_dim)
        .map(|i| half::f16::from_f32(((i as f32 % 23.0) - 11.0) * 0.009765625).to_bits())
        .collect::<Vec<_>>();
    let prior_v_bits = (0..prior_tokens * head_dim)
        .map(|i| half::f16::from_f32(((i as f32 % 19.0) - 9.0) * 0.0107421875).to_bits())
        .collect::<Vec<_>>();
    let attention_layer = Qwen35MtpDeviceVerifyAttentionMoeLayer {
        layer_index: 0,
        attn_norm: &attn_norm,
        q_q4k: &q,
        q_quant: 12,
        q_rows: 2 * head_dim,
        q_cols: hidden_dim,
        k_q4k: &k,
        k_quant: 12,
        k_rows: head_dim,
        k_cols: hidden_dim,
        v_q4k: &v,
        v_quant: 14,
        v_rows: head_dim,
        v_cols: hidden_dim,
        prior_k_bits: &prior_k_bits,
        prior_v_bits: &prior_v_bits,
        prior_tokens,
        o_q4k: &o,
        o_quant: GGML_Q4_K,
        o_rows: hidden_dim,
        o_cols: hidden_dim,
        q_norm: &q_norm,
        k_norm: &k_norm,
        post_attn_norm: &post_attn_norm,
        ffn_norm: &ffn_norm,
        ffn_gate_q4k: &[],
        ffn_gate_rows: 0,
        ffn_gate_cols: 0,
        ffn_up_q4k: &[],
        ffn_up_rows: 0,
        ffn_up_cols: 0,
        ffn_down: &[],
        ffn_down_quant: 12,
        ffn_down_rows: 0,
        ffn_down_cols: 0,
        router_w: &router,
        n_expert,
        n_expert_used,
        gate_all: &gate_all,
        up_all: &up_all,
        down_all: &down_all,
        down_quant: 12,
        shared_input_scale: &shared_input_scale,
        shared_gate: &shared_gate,
        shared_up: &shared_up,
        shared_down: &shared_down,
        shared_down_quant: 12,
        n_ff,
        n_embd: hidden_dim,
    };
    let attention_moe_layers = [attention_layer];
    let mut gdn_moe_layers = [];
    let layer_order = [];
    let request = Qwen35MtpDeviceVerifyRequest {
        verify_tokens: &verify_tokens,
        prefix_tokens: &prefix_tokens,
        pos_start: prior_tokens,
        hidden_dim,
        rope_dim: 256,
        rope_neox: true,
        rope_theta: 10000.0,
        include_bonus: false,
        token_embd_q4k: &token_embd,
        token_embd_quant: GGML_Q4_K,
        token_embd_rows,
        token_embd_cols: hidden_dim,
        layer_order: &layer_order,
        attention_moe_layers: &attention_moe_layers,
        gdn_moe_layers: &mut gdn_moe_layers,
        output_q6k: &output_q6k,
        output_quant: GGML_Q6_K,
        output_rows,
        output_cols: hidden_dim,
        output_norm: &output_norm,
        norm_eps: 1.0e-5,
    };

    let result = qwen35_mtp_device_verify_window(request).expect("attention-only device verify");

    assert_eq!(result.target_tokens.len(), verify_tokens.len());
    assert_eq!(
        result.mtp_hidden_rows.len(),
        verify_tokens.len() * hidden_dim
    );
    assert_eq!(result.prefix_states.len(), 1);
    assert_eq!(result.prefix_states[0].prefix_tokens, 1);
    assert!(result.prefix_states[0].layers.is_empty());
    assert_eq!(result.ssm_final_states.len(), 0);
    assert_eq!(result.attention_kv_states.len(), 1);
    assert_eq!(result.attention_kv_states[0].layer_idx, 0);
    assert_eq!(
        result.attention_kv_states[0].window_tokens,
        verify_tokens.len()
    );
    assert_eq!(result.attention_kv_states[0].kv_rows, head_dim);
    assert_eq!(
        result.attention_kv_states[0].k_bits.len(),
        verify_tokens.len() * head_dim
    );
    assert_eq!(
        result.attention_kv_states[0].v_bits.len(),
        verify_tokens.len() * head_dim
    );
}

#[test]
fn cuda_qwen35_mtp_device_verify_api_executes_layer_graph_before_unimplemented() {
    let _guard = runtime_test_lock();
    let verify_tokens = [10_u32, 11];
    let prefix_tokens = [1_usize];
    let hidden_dim = 512usize;
    let token_embd_rows = 16usize;
    let token_embd = make_test_q4k_weights(1, token_embd_rows, hidden_dim / 256, 197)
        .pop()
        .unwrap();
    let output_rows = 19usize;
    let output_q6k = make_test_q6k_weights(1, output_rows, hidden_dim / 256, 211)
        .pop()
        .unwrap();
    let output_norm = vec![1.0f32; hidden_dim];
    let attention_moe_layers = [];
    let mut delta_state = Vec::<f32>::new();
    let layer = Qwen35MtpDeviceVerifyGdnMoeLayer {
        layer_index: 0,
        attn_norm: &[],
        qkv_q4k: &[],
        qkv_quant: 12,
        qkv_rows: 0,
        qkv_cols: 0,
        gate_q4k: &[],
        gate_rows: 0,
        gate_cols: 0,
        alpha_q4k: &[],
        alpha_f32: &[],
        alpha_quant: GGML_Q4_K,
        alpha_rows: 0,
        alpha_cols: 0,
        beta_q4k: &[],
        beta_f32: &[],
        beta_quant: GGML_Q4_K,
        beta_rows: 0,
        beta_cols: 0,
        conv_state: &[],
        conv_kernel: &[],
        kernel_size: 0,
        dt_bias: &[],
        ssm_a: &[],
        num_k_heads: 0,
        num_v_heads: 0,
        head_k_dim: 0,
        head_v_dim: 0,
        delta_state: delta_state.as_mut_slice(),
        sync_delta_state_to_host: false,
        ssm_norm: &[],
        ssm_out_q4k: &[],
        ssm_out_quant: GGML_Q4_K,
        ssm_out_rows: 0,
        ssm_out_cols: 0,
        post_attn_norm: &[],
        router_w: &[],
        n_expert: 0,
        n_expert_used: 0,
        gate_all: &[],
        up_all: &[],
        down_all: &[],
        down_quant: 12,
        shared_input_scale: &[],
        shared_gate: &[],
        shared_up: &[],
        shared_down: &[],
        shared_down_quant: 12,
        n_ff: 0,
        n_embd: hidden_dim,
        ffn_gate_q4k: &[],
        ffn_gate_rows: 0,
        ffn_gate_cols: 0,
        ffn_up_q4k: &[],
        ffn_up_rows: 0,
        ffn_up_cols: 0,
        ffn_down: &[],
        ffn_down_quant: 12,
        ffn_down_rows: 0,
        ffn_down_cols: 0,
    };
    let mut gdn_moe_layers = [layer];
    let layer_order = [];
    let request = Qwen35MtpDeviceVerifyRequest {
        verify_tokens: &verify_tokens,
        prefix_tokens: &prefix_tokens,
        pos_start: 7,
        hidden_dim,
        rope_dim: 256,
        rope_neox: true,
        rope_theta: 10000.0,
        include_bonus: false,
        token_embd_q4k: &token_embd,
        token_embd_quant: GGML_Q4_K,
        token_embd_rows,
        token_embd_cols: hidden_dim,
        layer_order: &layer_order,
        attention_moe_layers: &attention_moe_layers,
        gdn_moe_layers: &mut gdn_moe_layers,
        output_q6k: &output_q6k,
        output_quant: GGML_Q6_K,
        output_rows,
        output_cols: hidden_dim,
        output_norm: &output_norm,
        norm_eps: 1.0e-5,
    };

    let err = match qwen35_mtp_device_verify_window(request) {
        Ok(_) => panic!("invalid MTP layer graph should not reach success"),
        Err(err) => err,
    };

    assert!(err.contains("GDN attn_norm length mismatch"));
    assert!(!err.contains("not implemented"));
}

#[test]
fn cuda_qwen35_mtp_device_verify_api_returns_gdn_prefix_states() {
    let _guard = runtime_test_lock();
    let _allow = EnvVarGuard::set("RNB_CUDA_ALLOW_EXPANDED_WEIGHT_CACHE", "1");
    let verify_tokens = [1_u32, 2];
    let prefix_tokens = [1_usize];
    let window_tokens = verify_tokens.len();
    let hidden_dim = 256usize;
    let num_k_heads = 1usize;
    let num_v_heads = 2usize;
    let head_k_dim = 4usize;
    let head_v_dim = 128usize;
    let d_inner = num_v_heads * head_v_dim;
    let conv_channels = num_k_heads * head_k_dim * 2 + d_inner;
    let kernel_size = 2usize;
    let n_ff = 256usize;
    let n_expert = 3usize;
    let n_expert_used = 2usize;
    let token_embd_rows = 8usize;
    let token_embd = make_test_q4k_weights(1, token_embd_rows, hidden_dim / 256, 787)
        .pop()
        .unwrap();
    let output_rows = 11usize;
    let output_q6k = make_test_q6k_weights(1, output_rows, hidden_dim / 256, 797)
        .pop()
        .unwrap();
    let output_norm = vec![1.0f32; hidden_dim];
    let attn_norm = (0..hidden_dim)
        .map(|i| 0.75 + (i % 13) as f32 * 0.0078125)
        .collect::<Vec<_>>();
    let qkv = make_test_q6k_weights(1, conv_channels, hidden_dim / 256, 809)
        .pop()
        .unwrap();
    let gate = make_test_q4k_weights(1, d_inner, hidden_dim / 256, 811)
        .pop()
        .unwrap();
    let alpha = make_test_q4k_weights(1, num_v_heads, hidden_dim / 256, 821)
        .pop()
        .unwrap();
    let beta = make_test_q4k_weights(1, num_v_heads, hidden_dim / 256, 823)
        .pop()
        .unwrap();
    let conv_state = (0..(kernel_size - 1) * conv_channels)
        .map(|i| ((i as f32 % 19.0) - 9.0) * 0.01171875)
        .collect::<Vec<_>>();
    let conv_kernel = (0..kernel_size * conv_channels)
        .map(|i| ((i as f32 % 23.0) - 11.0) * 0.009765625)
        .collect::<Vec<_>>();
    let dt_bias = [-0.25_f32, 0.125];
    let ssm_a = [-0.75_f32, -0.5];
    let mut delta_state = (0..num_v_heads * head_v_dim * head_k_dim)
        .map(|i| ((i as f32 % 29.0) - 14.0) * 0.005859375)
        .collect::<Vec<_>>();
    let ssm_norm = (0..head_v_dim)
        .map(|i| 0.5 + (i % 17) as f32 * 0.00390625)
        .collect::<Vec<_>>();
    let ssm_out = make_test_q4k_weights(1, hidden_dim, d_inner / 256, 827)
        .pop()
        .unwrap();
    let post_attn_norm = (0..hidden_dim)
        .map(|i| 0.625 + (i % 11) as f32 * 0.005859375)
        .collect::<Vec<_>>();
    let mut router = vec![0.0_f32; n_expert * hidden_dim];
    router[0] = 1.0;
    router[hidden_dim + 1] = 1.0;
    router[2 * hidden_dim] = 0.25;
    router[2 * hidden_dim + 1] = 0.5;
    let expert_gate = make_test_q4k_weights(n_expert, n_ff, hidden_dim / 256, 829);
    let expert_up = make_test_q4k_weights(n_expert, n_ff, hidden_dim / 256, 839);
    let expert_down = make_test_q4k_weights(n_expert, hidden_dim, n_ff / 256, 853);
    let gate_all = expert_gate.concat();
    let up_all = expert_up.concat();
    let down_all = expert_down.concat();
    let shared_input_scale = (0..hidden_dim)
        .map(|i| ((i as f32 % 7.0) - 3.0) * 0.0078125)
        .collect::<Vec<_>>();
    let shared_gate = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 857)
        .pop()
        .unwrap();
    let shared_up = make_test_q4k_weights(1, n_ff, hidden_dim / 256, 859)
        .pop()
        .unwrap();
    let shared_down = make_test_q4k_weights(1, hidden_dim, n_ff / 256, 863)
        .pop()
        .unwrap();
    let attention_moe_layers = [];
    let layer = Qwen35MtpDeviceVerifyGdnMoeLayer {
        layer_index: 5,
        attn_norm: &attn_norm,
        qkv_q4k: &qkv,
        qkv_quant: 14,
        qkv_rows: conv_channels,
        qkv_cols: hidden_dim,
        gate_q4k: &gate,
        gate_rows: d_inner,
        gate_cols: hidden_dim,
        alpha_q4k: &alpha,
        alpha_f32: &[],
        alpha_quant: GGML_Q4_K,
        alpha_rows: num_v_heads,
        alpha_cols: hidden_dim,
        beta_q4k: &beta,
        beta_f32: &[],
        beta_quant: GGML_Q4_K,
        beta_rows: num_v_heads,
        beta_cols: hidden_dim,
        conv_state: &conv_state,
        conv_kernel: &conv_kernel,
        kernel_size,
        dt_bias: &dt_bias,
        ssm_a: &ssm_a,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        delta_state: delta_state.as_mut_slice(),
        sync_delta_state_to_host: true,
        ssm_norm: &ssm_norm,
        ssm_out_q4k: &ssm_out,
        ssm_out_quant: GGML_Q4_K,
        ssm_out_rows: hidden_dim,
        ssm_out_cols: d_inner,
        post_attn_norm: &post_attn_norm,
        router_w: &router,
        n_expert,
        n_expert_used,
        gate_all: &gate_all,
        up_all: &up_all,
        down_all: &down_all,
        down_quant: 12,
        shared_input_scale: &shared_input_scale,
        shared_gate: &shared_gate,
        shared_up: &shared_up,
        shared_down: &shared_down,
        shared_down_quant: 12,
        n_ff,
        n_embd: hidden_dim,
        ffn_gate_q4k: &[],
        ffn_gate_rows: 0,
        ffn_gate_cols: 0,
        ffn_up_q4k: &[],
        ffn_up_rows: 0,
        ffn_up_cols: 0,
        ffn_down: &[],
        ffn_down_quant: 12,
        ffn_down_rows: 0,
        ffn_down_cols: 0,
    };
    let mut gdn_moe_layers = [layer];
    let layer_order = [Qwen35MtpDeviceVerifyLayerKind::GdnMoe(0)];
    let request = Qwen35MtpDeviceVerifyRequest {
        verify_tokens: &verify_tokens,
        prefix_tokens: &prefix_tokens,
        pos_start: 7,
        hidden_dim,
        rope_dim: 256,
        rope_neox: true,
        rope_theta: 10000.0,
        include_bonus: false,
        token_embd_q4k: &token_embd,
        token_embd_quant: GGML_Q4_K,
        token_embd_rows,
        token_embd_cols: hidden_dim,
        layer_order: &layer_order,
        attention_moe_layers: &attention_moe_layers,
        gdn_moe_layers: &mut gdn_moe_layers,
        output_q6k: &output_q6k,
        output_quant: GGML_Q6_K,
        output_rows,
        output_cols: hidden_dim,
        output_norm: &output_norm,
        norm_eps: 1.0e-5,
    };

    let result = qwen35_mtp_device_verify_window(request).expect("GDN-only device verify result");

    assert_eq!(result.target_tokens.len(), window_tokens);
    assert_eq!(result.hidden_dim, hidden_dim);
    assert_eq!(result.mtp_hidden_rows.len(), window_tokens * hidden_dim);
    assert_eq!(result.prefix_states.len(), 1);
    assert_eq!(result.prefix_states[0].prefix_tokens, 1);
    assert_eq!(result.prefix_states[0].layers.len(), 1);
    assert_eq!(result.prefix_states[0].layers[0].layer_idx, 5);
    assert_eq!(
        result.prefix_states[0].layers[0].conv_state.len(),
        (kernel_size - 1) * conv_channels
    );
    assert_eq!(result.ssm_final_states.len(), 1);
    assert_eq!(result.ssm_final_states[0].layer_idx, 5);
    assert_eq!(
        result.ssm_final_states[0].conv_state.len(),
        (kernel_size - 1) * conv_channels
    );
    let snapshot = result.prefix_states[0].layers[0]
        .resident_delta_snapshot
        .as_ref()
        .expect("resident delta snapshot");
    assert!(restore_delta_state_cache(&mut delta_state, snapshot).expect("restore snapshot"));
    assert!(sync_delta_state_cache(&mut delta_state).expect("sync restored snapshot"));
}
