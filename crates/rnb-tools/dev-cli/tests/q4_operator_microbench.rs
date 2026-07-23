//! mt91 — operator microbench unit + env-gated integration tests.

use rnb_dev_tools::q4_operator_microbench::{run_four_paths, OperatorKind, OperatorMicrobenchArgs};
use std::path::PathBuf;

#[test]
fn operator_kind_parses_known_names() {
    assert_eq!(OperatorKind::parse("o_proj").unwrap(), OperatorKind::OProj);
    assert_eq!(
        OperatorKind::parse("mlp_gate").unwrap(),
        OperatorKind::MlpGate
    );
    assert_eq!(OperatorKind::parse("mlp_up").unwrap(), OperatorKind::MlpUp);
    assert_eq!(
        OperatorKind::parse("mlp_down").unwrap(),
        OperatorKind::MlpDown
    );
    assert_eq!(OperatorKind::parse("v_proj").unwrap(), OperatorKind::VProj);
    assert!(OperatorKind::parse("garbage").is_err());
}

#[test]
fn args_validates_paths() {
    let args = OperatorMicrobenchArgs {
        gguf_path: PathBuf::from("/nonexistent.gguf"),
        layer: 17,
        operator: OperatorKind::OProj,
        input_bin: PathBuf::from("/nonexistent_input.bin"),
        output_dir: PathBuf::from("/tmp/mt91_microbench_test"),
    };
    let result = run_four_paths(&args);
    assert!(result.is_err(), "expected error for missing files");
}

#[test]
fn run_four_paths_outputs_three_files() {
    use std::fs;

    let gguf = std::env::var("MT91_TEST_GGUF").ok().map(PathBuf::from);
    let input = std::env::var("MT91_TEST_INPUT_BIN").ok().map(PathBuf::from);
    let Some((gguf, input)) = gguf.zip(input) else {
        eprintln!("skip: set MT91_TEST_GGUF + MT91_TEST_INPUT_BIN");
        return;
    };
    let layer: usize = std::env::var("MT91_TEST_LAYER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(17);
    let op_name = std::env::var("MT91_TEST_OPERATOR").unwrap_or_else(|_| "o_proj".to_string());
    let operator = OperatorKind::parse(&op_name).expect("invalid MT91_TEST_OPERATOR");

    let out_dir = PathBuf::from("/tmp/mt91_microbench_integration");
    let _ = fs::remove_dir_all(&out_dir);

    let args = OperatorMicrobenchArgs {
        gguf_path: gguf,
        layer,
        operator,
        input_bin: input,
        output_dir: out_dir.clone(),
    };
    run_four_paths(&args).expect("4-path run failed");

    let stem = format!("layer_{:03}_{}", layer, op_name);
    let prod_path = out_dir.join(format!("{stem}_path_prod.bin"));
    let generic_path = out_dir.join(format!("{stem}_path_generic.bin"));
    let f64_path = out_dir.join(format!("{stem}_path_f64.bin"));
    assert!(prod_path.exists());
    assert!(generic_path.exists());
    assert!(f64_path.exists());

    // Sanity: all three path outputs must be the same byte length (== rows * 4).
    // This catches axis-swap regressions where `rows` and `cols` get flipped
    // when reading float_shape from the GGUF loader. If `MT91_TEST_EXPECTED_ROWS`
    // is set, also assert the absolute output dim — primary axis-swap guard.
    let prod_bytes = fs::metadata(&prod_path).expect("stat prod").len();
    let generic_bytes = fs::metadata(&generic_path).expect("stat generic").len();
    let f64_bytes = fs::metadata(&f64_path).expect("stat f64").len();
    assert_eq!(
        prod_bytes, generic_bytes,
        "prod vs generic output size mismatch (axis swap?)"
    );
    assert_eq!(
        prod_bytes, f64_bytes,
        "prod vs f64 output size mismatch (axis swap?)"
    );
    assert!(
        prod_bytes > 0 && prod_bytes % 4 == 0,
        "prod output not f32-aligned"
    );

    if let Ok(expected) = std::env::var("MT91_TEST_EXPECTED_ROWS") {
        let expected_rows: u64 = expected.parse().expect("MT91_TEST_EXPECTED_ROWS not int");
        assert_eq!(
            prod_bytes,
            expected_rows * 4,
            "prod output dim mismatch: expected {expected_rows} rows (={} bytes), got {prod_bytes} bytes",
            expected_rows * 4
        );
    }
}
