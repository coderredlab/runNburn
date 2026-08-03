use super::{cuda_driver_unavailable_for_test, runtime_test_lock};

fn constant_mxfp4_matrix(rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0);
    let mut matrix = Vec::with_capacity(rows * cols / 32 * 17);
    for _ in 0..rows * cols / 32 {
        matrix.push(128);
        matrix.extend([0x11; 16]);
    }
    matrix
}

fn patterned_mxfp4_matrix(rows: usize, cols: usize, seed: u8) -> Vec<u8> {
    assert_eq!(cols % 32, 0);
    let mut matrix = Vec::with_capacity(rows * cols / 32 * 17);
    for row in 0..rows {
        for block in 0..cols / 32 {
            matrix.push(126 + ((row + block + seed as usize) % 3) as u8);
            for packed in 0..16 {
                let low = ((row * 3 + block * 5 + packed + seed as usize) % 15 + 1) as u8;
                let high = ((row * 7 + block * 11 + packed * 3 + seed as usize) % 15 + 1) as u8;
                matrix.push(low | (high << 4));
            }
        }
    }
    matrix
}

fn mxfp4_value(block: &[u8], index: usize) -> f32 {
    const VALUES: [f32; 16] = [
        0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
    ];
    let encoded = block[0];
    let bits = if encoded < 2 {
        0x0020_0000u32 << encoded
    } else {
        ((encoded as u32) - 1) << 23
    };
    let packed = block[1 + (index & 15)];
    let quant = if index < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    f32::from_bits(bits) * VALUES[quant as usize]
}

fn mxfp4_matvec(matrix: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    let row_bytes = cols / 32 * 17;
    (0..rows)
        .map(|row| {
            let row = &matrix[row * row_bytes..(row + 1) * row_bytes];
            row.chunks_exact(17)
                .zip(input.chunks_exact(32))
                .map(|(block, input)| {
                    (0..32)
                        .map(|index| mxfp4_value(block, index) * input[index])
                        .sum::<f32>()
                })
                .sum()
        })
        .collect()
}

#[test]
fn mxfp4_selected_moe_matches_constant_weight_oracle() {
    let _guard = runtime_test_lock();
    let n_embd = 256;
    let n_ff = 256;
    let gate = constant_mxfp4_matrix(n_ff, n_embd);
    let up = constant_mxfp4_matrix(n_ff, n_embd);
    let down = constant_mxfp4_matrix(n_embd, n_ff);
    let input = (0..n_embd)
        .map(|index| (index % 7) as f32 * 0.001)
        .collect::<Vec<_>>();
    let route = 0.5f32;
    let activation_limit = 7.0f32;

    let actual = match crate::runtime::mxfp4_sparse_experts_by_token_clamped_swiglu(
        &[gate.as_slice()],
        &[up.as_slice()],
        &[down.as_slice()],
        &[route],
        &[0],
        1,
        n_ff,
        n_embd,
        &input,
        activation_limit,
    ) {
        Ok(actual) => actual,
        Err(error) if cuda_driver_unavailable_for_test(&error) => {
            eprintln!("skipping MXFP4 DSpark CUDA test: {error}");
            return;
        }
        Err(error) => panic!("MXFP4 DSpark CUDA execution failed: {error}"),
    };

    let projection = input.iter().sum::<f32>();
    let gate_value = projection.min(activation_limit);
    let up_value = projection.clamp(-activation_limit, activation_limit);
    let activated = gate_value / (1.0 + (-gate_value).exp()) * up_value;
    let expected = activated * n_ff as f32 * route;
    assert_eq!(actual.len(), n_embd);
    for (index, value) in actual.into_iter().enumerate() {
        let tolerance = 1e-3f32.max(expected.abs() * 1e-5);
        assert!(
            (value - expected).abs() <= tolerance,
            "row {index}: actual={value} expected={expected} tolerance={tolerance}"
        );
    }
}

#[test]
fn mxfp4_selected_moe_matches_patterned_weight_oracle() {
    let _guard = runtime_test_lock();
    let n_embd = 256;
    let n_ff = 256;
    let gates = [
        patterned_mxfp4_matrix(n_ff, n_embd, 1),
        patterned_mxfp4_matrix(n_ff, n_embd, 2),
    ];
    let ups = [
        patterned_mxfp4_matrix(n_ff, n_embd, 3),
        patterned_mxfp4_matrix(n_ff, n_embd, 4),
    ];
    let downs = [
        patterned_mxfp4_matrix(n_embd, n_ff, 5),
        patterned_mxfp4_matrix(n_embd, n_ff, 6),
    ];
    let input = (0..2 * n_embd)
        .map(|index| ((index * 17 % 29) as f32 - 14.0) * 0.0005)
        .collect::<Vec<_>>();
    let routes = [0.375f32, 0.625f32];
    let activation_limit = 7.0f32;
    let gate_refs = gates.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = ups.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let down_refs = downs.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let actual = match crate::runtime::mxfp4_sparse_experts_by_token_clamped_swiglu(
        &gate_refs,
        &up_refs,
        &down_refs,
        &routes,
        &[0, 1],
        2,
        n_ff,
        n_embd,
        &input,
        activation_limit,
    ) {
        Ok(actual) => actual,
        Err(error) if cuda_driver_unavailable_for_test(&error) => {
            eprintln!("skipping MXFP4 DSpark CUDA test: {error}");
            return;
        }
        Err(error) => panic!("MXFP4 DSpark CUDA execution failed: {error}"),
    };

    let mut expected = Vec::with_capacity(2 * n_embd);
    for token in 0..2 {
        let input = &input[token * n_embd..(token + 1) * n_embd];
        let mut gate = mxfp4_matvec(&gates[token], n_ff, n_embd, input);
        let up = mxfp4_matvec(&ups[token], n_ff, n_embd, input);
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = gate.min(activation_limit) / (1.0 + (-*gate).exp())
                * up.clamp(-activation_limit, activation_limit);
        }
        expected.extend(
            mxfp4_matvec(&downs[token], n_embd, n_ff, &gate)
                .into_iter()
                .map(|value| value * routes[token]),
        );
    }

    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let tolerance = 2e-3f32.max(expected.abs() * 2e-5);
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

#[test]
fn mxfp4_sparse_moe_rejects_mixed_quant_abis_before_cuda_init() {
    let empty = &[][..];
    let input = vec![0.0f32; 256];
    for (gate_quant, down_quant) in [(39, 18), (16, 39)] {
        let error = crate::runtime::glm_sparse_experts_iq_by_token(
            &[empty],
            &[empty],
            &[empty],
            gate_quant,
            down_quant,
            None,
            false,
            &[1.0],
            &[0],
            1,
            256,
            256,
            &input,
        )
        .expect_err("mixed MXFP4/IQ quant ABI must be rejected");
        assert!(
            error.contains("requires gate/up and down quant code 39 together"),
            "unexpected mixed-quant error: {error}"
        );
    }
}
