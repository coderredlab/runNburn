use super::*;

#[test]
fn selected_sparse_megakernel_matches_rank_ordered_cpu_reference() {
    let _guard = runtime_test_lock();
    let tokens = 2usize;
    let hidden_dim = 256usize;
    let n_ff = 32usize;
    let n_expert = 4usize;
    let top_k = 2usize;
    let q4_blocks = hidden_dim / 256;

    let normalized = (0..tokens * hidden_dim)
        .map(|index| ((index as f32 % 29.0) - 14.0) * 0.0078125)
        .collect::<Vec<_>>();
    let expert_ids = vec![2u32, 0, 1, 3];
    let route_weights = vec![0.6875f32, 0.3125, 0.59375, 0.40625];
    let down_scale = vec![0.75f32, 1.125, 0.875, 1.25];
    let gate_up_per_expert = make_test_q4k_weights(n_expert, n_ff * 2, q4_blocks, 1709);
    let gate_up_weights = gate_up_per_expert
        .iter()
        .flat_map(|weights| weights.iter().copied())
        .collect::<Vec<_>>();
    let down_weights = make_test_q5_basic_weights(n_expert * hidden_dim, n_ff, 24, true, 1783);

    let actual = match gemma_mtp2_selected_sparse_for_test(
        &normalized,
        &expert_ids,
        &route_weights,
        &gate_up_weights,
        &down_weights,
        &down_scale,
        tokens,
        hidden_dim,
        n_ff,
        n_expert,
        top_k,
        7,
        None,
    ) {
        Ok(result) => result.output,
        Err(error) if cuda_driver_unavailable_for_test(&error) => {
            eprintln!("skipping Gemma MTP2 megakernel test: {error}");
            return;
        }
        Err(error) => panic!("Gemma MTP2 megakernel execution failed: {error}"),
    };

    let gate_up_row_bytes = q4_blocks * 144;
    let gate_up_expert_bytes = n_ff * 2 * gate_up_row_bytes;
    let down_row_bytes = (n_ff / 32) * 24;
    let down_expert_bytes = hidden_dim * down_row_bytes;
    let mut expected = vec![0.0f32; tokens * hidden_dim];
    for token in 0..tokens {
        let input = &normalized[token * hidden_dim..(token + 1) * hidden_dim];
        for rank in 0..top_k {
            let slot = token * top_k + rank;
            let expert = expert_ids[slot] as usize;
            let gate_up = cpu_q4k_gemv_rows(
                &gate_up_weights
                    [expert * gate_up_expert_bytes..(expert + 1) * gate_up_expert_bytes],
                n_ff * 2,
                q4_blocks,
                input,
            );
            let activation = (0..n_ff)
                .map(|row| {
                    let gate = gate_up[row];
                    let gate3 = gate * gate * gate;
                    let gelu = 0.5
                        * gate
                        * (1.0 + (0.7978845608028654 * (gate + 0.044715 * gate3)).tanh());
                    gelu * gate_up[n_ff + row]
                })
                .collect::<Vec<_>>();
            let down = cpu_q5_basic_rows(
                &down_weights[expert * down_expert_bytes..(expert + 1) * down_expert_bytes],
                hidden_dim,
                n_ff,
                24,
                true,
                &activation,
            );
            let scale = route_weights[slot] * down_scale[expert];
            for row in 0..hidden_dim {
                expected[token * hidden_dim + row] += down[row] * scale;
            }
        }
    }

    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let tolerance = 2.0e-4f32.max(expected.abs() * 2.0e-5);
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "Gemma MTP2 output mismatch at {index}: actual={actual} expected={expected} diff={diff} tolerance={tolerance}"
        );
    }
}

#[test]
fn selected_sparse_megakernel_matches_product_stages_and_rank_reduce() {
    let _guard = runtime_test_lock();
    let tokens = 2usize;
    let hidden_dim = 2816usize;
    let n_ff = 704usize;
    let n_expert = 128usize;
    let top_k = 8usize;
    let normalized = (0..tokens * hidden_dim)
        .map(|index| ((index as f32 % 41.0) - 20.0) * 0.015625)
        .collect::<Vec<_>>();
    let expert_ids = vec![
        127u32, 2, 97, 1, 109, 44, 112, 66, 3, 2, 111, 127, 88, 55, 0, 113,
    ];
    let route_weights = (0..tokens * top_k)
        .map(|index| 0.05f32 + index as f32 * 0.0073)
        .collect::<Vec<_>>();
    let down_scale = (0..n_expert)
        .map(|expert| 0.713f32 + expert as f32 * 0.013)
        .collect::<Vec<_>>();
    let gate_up_weights = make_test_q4k_weights(n_expert, n_ff * 2, hidden_dim / 256, 1877)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let down_weights = make_test_q5_basic_weights(n_expert * hidden_dim, n_ff, 24, true, 1901);

    let candidate = match gemma_mtp2_selected_sparse_for_test(
        &normalized,
        &expert_ids,
        &route_weights,
        &gate_up_weights,
        &down_weights,
        &down_scale,
        tokens,
        hidden_dim,
        n_ff,
        n_expert,
        top_k,
        7,
        None,
    ) {
        Ok(result) => result,
        Err(error) if cuda_driver_unavailable_for_test(&error) => {
            eprintln!("skipping Gemma MTP2 product cross-check: {error}");
            return;
        }
        Err(error) => panic!("Gemma MTP2 megakernel execution failed: {error}"),
    };

    let mut routes = expert_ids
        .iter()
        .copied()
        .zip(route_weights.iter().copied())
        .enumerate()
        .map(|(slot, (expert, weight))| {
            let token = (slot / top_k) as u32;
            (expert, token, weight * down_scale[expert as usize])
        })
        .collect::<Vec<_>>();
    routes.sort_unstable_by_key(|&(expert, token, _)| (expert, token));
    let grouped_experts = routes.iter().map(|route| route.0).collect::<Vec<_>>();
    let grouped_tokens = routes.iter().map(|route| route.1).collect::<Vec<_>>();
    let grouped_weights = routes.iter().map(|route| route.2).collect::<Vec<_>>();
    let expected = gemma4_moe_gelu_selected(
        &gate_up_weights,
        &down_weights,
        7,
        n_expert,
        n_ff,
        hidden_dim,
        tokens,
        &grouped_experts,
        &grouped_tokens,
        &grouped_weights,
        &normalized,
    )
    .expect("Gemma product grouped path failed");

    let mut rank_reduced = vec![0.0f32; tokens * hidden_dim];
    for token in 0..tokens {
        for rank in 0..top_k {
            let slot = token * top_k + rank;
            for row in 0..hidden_dim {
                rank_reduced[token * hidden_dim + row] +=
                    candidate.rank_output[slot * hidden_dim + row];
            }
        }
    }
    assert_eq!(
        candidate.output, rank_reduced,
        "Gemma MTP2 output must reduce expert contributions in route-rank order"
    );

    let mut expert_reduced = vec![0.0f32; tokens * hidden_dim];
    for token in 0..tokens {
        let mut ranks = (0..top_k).collect::<Vec<_>>();
        ranks.sort_unstable_by_key(|&rank| expert_ids[token * top_k + rank]);
        for rank in ranks {
            let slot = token * top_k + rank;
            for row in 0..hidden_dim {
                expert_reduced[token * hidden_dim + row] +=
                    candidate.rank_output[slot * hidden_dim + row];
            }
        }
    }
    assert_eq!(
        expert_reduced, expected,
        "Gemma MTP2 per-route stages diverge from product grouped path"
    );
}

#[test]
fn selected_sparse_megakernel_matches_product_one_expert_stages() {
    let _guard = runtime_test_lock();
    let tokens = 1usize;
    let hidden_dim = 2816usize;
    let n_ff = 704usize;
    let normalized = (0..tokens * hidden_dim)
        .map(|index| ((index as f32 % 41.0) - 20.0) * 0.015625)
        .collect::<Vec<_>>();
    let gate_up_weights = make_test_q4k_weights(1, n_ff * 2, hidden_dim / 256, 1973)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let down_weights = make_test_q5_basic_weights(hidden_dim, n_ff, 24, true, 1999);

    let candidate = match gemma_mtp2_selected_sparse_for_test(
        &normalized,
        &[0],
        &[1.0],
        &gate_up_weights,
        &down_weights,
        &[1.0],
        tokens,
        hidden_dim,
        n_ff,
        1,
        1,
        7,
        None,
    ) {
        Ok(result) => result,
        Err(error) if cuda_driver_unavailable_for_test(&error) => {
            eprintln!("skipping Gemma MTP2 stage cross-check: {error}");
            return;
        }
        Err(error) => panic!("Gemma MTP2 megakernel execution failed: {error}"),
    };
    let product = gemma_product_ffn_for_test(
        &normalized,
        &gate_up_weights,
        &down_weights,
        7,
        n_ff,
        hidden_dim,
    )
    .expect("Gemma product FFN failed");
    let (up_max, up_mean) = max_and_mean_abs_delta(&candidate.up, &product.up);
    let (activation_max, activation_mean) =
        max_and_mean_abs_delta(&candidate.activation, &product.activation);
    let (down_max, down_mean) = max_and_mean_abs_delta(&candidate.rank_output, &product.output);
    eprintln!(
        "Gemma MTP2 stage delta: up max={up_max:.8} mean={up_mean:.8}; activation max={activation_max:.8} mean={activation_mean:.8}; down max={down_max:.8} mean={down_mean:.8}"
    );
    assert_eq!(
        (up_max, up_mean),
        (0.0, 0.0),
        "Gemma MTP2 up projection diverges"
    );
    assert_eq!(
        (activation_max, activation_mean),
        (0.0, 0.0),
        "Gemma MTP2 activation diverges"
    );
    assert_eq!(
        (down_max, down_mean),
        (0.0, 0.0),
        "Gemma MTP2 down projection diverges"
    );
}

#[test]
fn selected_sparse_megakernel_matches_product_q8_0_down() {
    let _guard = runtime_test_lock();
    let hidden_dim = 256usize;
    let n_ff = 32usize;
    let normalized = (0..hidden_dim)
        .map(|index| ((index as f32 % 37.0) - 18.0) * 0.01171875)
        .collect::<Vec<_>>();
    let gate_up_weights = make_test_q4k_weights(1, n_ff * 2, 1, 2027)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let down_weights = make_test_q8_0_weights(hidden_dim, n_ff, 2053);

    let candidate = match gemma_mtp2_selected_sparse_for_test(
        &normalized,
        &[0],
        &[1.0],
        &gate_up_weights,
        &down_weights,
        &[1.0],
        1,
        hidden_dim,
        n_ff,
        1,
        1,
        8,
        None,
    ) {
        Ok(result) => result,
        Err(error) if cuda_driver_unavailable_for_test(&error) => {
            eprintln!("skipping Gemma MTP2 Q8_0 cross-check: {error}");
            return;
        }
        Err(error) => panic!("Gemma MTP2 Q8_0 megakernel execution failed: {error}"),
    };
    let product = gemma_product_ffn_for_test(
        &normalized,
        &gate_up_weights,
        &down_weights,
        8,
        n_ff,
        hidden_dim,
    )
    .expect("Gemma Q8_0 product FFN failed");

    assert_eq!(candidate.up, product.up);
    assert_eq!(candidate.activation, product.activation);
    assert_eq!(candidate.rank_output, product.output);
}
#[test]
fn finalize_megakernel_matches_three_norms_and_residual_contract() {
    let _guard = runtime_test_lock();
    let tokens = 2usize;
    let hidden_dim = 256usize;
    let n_ff = 32usize;
    let normalized = (0..tokens * hidden_dim)
        .map(|index| ((index as f32 % 31.0) - 15.0) * 0.0125)
        .collect::<Vec<_>>();
    let gate_up_weights = make_test_q4k_weights(1, n_ff * 2, 1, 2063)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let down_weights = make_test_q5_basic_weights(hidden_dim, n_ff, 24, true, 2081);
    let shared_raw = (0..tokens * hidden_dim)
        .map(|index| ((index as f32 % 23.0) - 11.0) * 0.03125)
        .collect::<Vec<_>>();
    let residual = (0..tokens * hidden_dim)
        .map(|index| ((index as f32 % 19.0) - 9.0) * 0.0625)
        .collect::<Vec<_>>();
    let post_norm_1 = (0..hidden_dim)
        .map(|index| (index as f32 % 7.0) * 0.003)
        .collect::<Vec<_>>();
    let post_norm_2 = (0..hidden_dim)
        .map(|index| (index as f32 % 5.0) * -0.002)
        .collect::<Vec<_>>();
    let common_post_norm = (0..hidden_dim)
        .map(|index| (index as f32 % 11.0) * 0.0015)
        .collect::<Vec<_>>();
    let norm_eps = 1.0e-6f32;

    let base = match gemma_mtp2_selected_sparse_for_test(
        &normalized,
        &[0, 0],
        &[1.0, 1.0],
        &gate_up_weights,
        &down_weights,
        &[1.0],
        tokens,
        hidden_dim,
        n_ff,
        1,
        1,
        7,
        None,
    ) {
        Ok(result) => result,
        Err(error) if cuda_driver_unavailable_for_test(&error) => {
            eprintln!("skipping Gemma MTP2 finalize cross-check: {error}");
            return;
        }
        Err(error) => panic!("Gemma MTP2 base execution failed: {error}"),
    };
    let fused = gemma_mtp2_selected_sparse_for_test(
        &normalized,
        &[0, 0],
        &[1.0, 1.0],
        &gate_up_weights,
        &down_weights,
        &[1.0],
        tokens,
        hidden_dim,
        n_ff,
        1,
        1,
        7,
        Some(GemmaMtp2FinalizeTestInput {
            residual: &residual,
            shared_raw: &shared_raw,
            post_norm_1: &post_norm_1,
            post_norm_2: &post_norm_2,
            common_post_norm: &common_post_norm,
            norm_eps,
            unit_offset: true,
        }),
    )
    .expect("Gemma MTP2 finalize execution failed");
    let actual = fused
        .finalized_residual
        .expect("finalize request must return residual");

    let mut expected = residual;
    for token in 0..tokens {
        let start = token * hidden_dim;
        let end = start + hidden_dim;
        let shared = &shared_raw[start..end];
        let sparse = &base.output[start..end];
        let shared_rrms = 1.0
            / (shared.iter().map(|value| value * value).sum::<f32>() / hidden_dim as f32
                + norm_eps)
                .sqrt();
        let sparse_rrms = 1.0
            / (sparse.iter().map(|value| value * value).sum::<f32>() / hidden_dim as f32
                + norm_eps)
                .sqrt();
        let combined = (0..hidden_dim)
            .map(|row| {
                shared[row] * shared_rrms * (1.0 + post_norm_1[row])
                    + sparse[row] * sparse_rrms * (1.0 + post_norm_2[row])
            })
            .collect::<Vec<_>>();
        let combined_rrms = 1.0
            / (combined.iter().map(|value| value * value).sum::<f32>() / hidden_dim as f32
                + norm_eps)
                .sqrt();
        for row in 0..hidden_dim {
            expected[start + row] += combined[row] * combined_rrms * (1.0 + common_post_norm[row]);
        }
    }

    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let tolerance = 2.0e-5f32.max(expected.abs() * 2.0e-5);
        assert!(
            (actual - expected).abs() <= tolerance,
            "Gemma MTP2 finalize mismatch at {index}: actual={actual} expected={expected}"
        );
    }
}

fn max_and_mean_abs_delta(lhs: &[f32], rhs: &[f32]) -> (f32, f64) {
    assert_eq!(lhs.len(), rhs.len());
    let (mut max_abs, mut sum_abs) = (0.0f32, 0.0f64);
    for (&lhs, &rhs) in lhs.iter().zip(rhs.iter()) {
        let diff = (lhs - rhs).abs();
        max_abs = max_abs.max(diff);
        sum_abs += f64::from(diff);
    }
    (max_abs, sum_abs / lhs.len() as f64)
}
