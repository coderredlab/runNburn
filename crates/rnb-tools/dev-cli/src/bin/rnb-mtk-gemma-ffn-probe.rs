#[cfg(not(feature = "mediatek"))]
fn main() {
    eprintln!("rnb-mtk-gemma-ffn-probe requires --features mediatek");
    std::process::exit(2);
}

#[cfg(feature = "mediatek")]
fn main() {
    std::process::exit(app::run());
}

#[cfg(feature = "mediatek")]
mod app {
    use std::path::PathBuf;

    use rnb_dev_tools::mtk_gguf_mlp::{
        extract_gated_gelu_ffn_from_loaded_model, MtkGatedGeluFfnConfig,
    };
    use rnb_runtime::mediatek::{
        compile_gated_gelu_ffn_f32_batched, compile_gated_gelu_ffn_f32_batched_with,
        probe_gated_gelu_ffn_f32, probe_gated_gelu_ffn_f32_batched,
        run_compiled_gated_gelu_ffn_f32_batched, BatchedCompileOptions,
        MediaTekGatedGeluFfnBatchedCompilation, MediaTekProbeError,
        ProbeGatedGeluFfnF32BatchedRequest, ProbeGatedGeluFfnF32Request,
    };

    const SUCCESS_SYNTHETIC: &str = "MTK_NNAPI_GEMMA_FFN_SYNTHETIC_OK";
    const SUCCESS_GGUF_LAYER: &str = "MTK_NNAPI_GEMMA_FFN_GGUF_LAYER_OK";
    const SUCCESS_GGUF_CO_RESIDENT: &str = "MTK_NNAPI_GEMMA_FFN_CO_RESIDENT_OK";
    const DEFAULT_CO_RESIDENT_BATCH: usize = 128;

    pub(super) fn run() -> i32 {
        match Args::parse(std::env::args().skip(1)) {
            Ok(args) if args.help => {
                print_help();
                0
            }
            Ok(args) => run_with_args(args),
            Err(err) => {
                eprintln!("{err}");
                eprintln!("run with --help for usage");
                2
            }
        }
    }

    fn run_with_args(args: Args) -> i32 {
        if args.synthetic {
            return run_synthetic(&args);
        }
        if args.gguf.is_some() {
            if let Some(co_resident_layers) = args.co_resident {
                return run_gguf_layer_co_resident(&args, co_resident_layers);
            }
            return if let Some(batch) = args.batch {
                run_gguf_layer_batched(&args, batch)
            } else {
                run_gguf_layer(&args)
            };
        }
        eprintln!("one mode is required: --synthetic or --gguf PATH");
        2
    }

    fn run_synthetic(args: &Args) -> i32 {
        let synthetic = SyntheticCase::new();
        let reference = synthetic.cpu_reference();
        let request = ProbeGatedGeluFfnF32Request {
            device_name_substring: args.device.clone(),
            input_size: synthetic.input_size,
            ffn_inner_size: synthetic.ffn_inner_size,
            output_size: synthetic.output_size,
            gate_weight: &synthetic.gate_weight,
            up_weight: &synthetic.up_weight,
            down_weight: &synthetic.down_weight,
            input: &synthetic.input,
        };
        let output = match probe_gated_gelu_ffn_f32(request) {
            Ok(output) => output,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_SYNTHETIC_NO_GO error={err}");
                return match err {
                    MediaTekProbeError::UnsupportedPlatform => 1,
                    _ => 2,
                };
            }
        };

        let metrics = match Metrics::new(&reference, &output.output) {
            Ok(metrics) => metrics,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_SYNTHETIC_NO_GO error={err}");
                return 2;
            }
        };
        let finite_ok = metrics.finite_count == synthetic.output_size;
        let error_ok = metrics.max_abs_error <= args.tolerance;
        let support_ok = output.supported_ops.iter().all(|(_, supported)| *supported);
        let accelerator_ok =
            output.chosen_device_type == rnb_backend_mediatek::MEDIATEK_NNAPI_DEVICE_ACCELERATOR;

        println!("mediatek_gemma_ffn_probe=synthetic");
        println!("chosen_device.name={}", output.chosen_device_name);
        println!("chosen_device.type={}", output.chosen_device_type);
        println!(
            "chosen_device.feature_level={}",
            output.chosen_device_feature_level
        );
        println!("chosen_device.version={}", output.chosen_device_version);
        for (name, supported) in &output.supported_ops {
            println!("supported_ops.{name}={supported}");
        }
        println!("duration_hardware_ns={:?}", output.duration_hardware_ns);
        println!("duration_driver_ns={:?}", output.duration_driver_ns);
        println!("finite_count={}", metrics.finite_count);
        println!("output_len={}", output.output.len());
        println!("max_abs_error={:.9}", metrics.max_abs_error);

        if finite_ok && error_ok && support_ok && accelerator_ok {
            println!("{SUCCESS_SYNTHETIC}");
            0
        } else {
            eprintln!(
                "MTK_NNAPI_GEMMA_FFN_SYNTHETIC_NO_GO finite_ok={finite_ok} \
                 error_ok={error_ok} support_ok={support_ok} accelerator_ok={accelerator_ok}"
            );
            1
        }
    }

    fn run_gguf_layer(args: &Args) -> i32 {
        let Some(gguf) = args.gguf.as_ref() else {
            eprintln!("one mode is required: --synthetic or --gguf PATH");
            return 2;
        };
        if args.prompt_file.is_some() {
            eprintln!("--prompt-file is reserved for the token smoke path; GGUF layer probe uses --input-row");
            return 2;
        }

        eprintln!("[load] {}", gguf.display());
        let model = match rnb_loader::load_model(gguf) {
            Ok(model) => model,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_GGUF_LAYER_NO_GO load_model failed: {err:?}");
                return 1;
            }
        };
        let config = MtkGatedGeluFfnConfig {
            layer: args.layer.unwrap_or(0),
            input_row: args.input_row,
            input_scale: args.input_scale,
            ..MtkGatedGeluFfnConfig::default()
        };
        let extracted = match extract_gated_gelu_ffn_from_loaded_model(&model, &config) {
            Ok(extracted) => extracted,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_GGUF_LAYER_NO_GO extract failed: {err}");
                return 1;
            }
        };
        let request = ProbeGatedGeluFfnF32Request {
            device_name_substring: args.device.clone(),
            input_size: extracted.payload.input_size,
            ffn_inner_size: extracted.payload.ffn_inner_size,
            output_size: extracted.payload.output_size,
            gate_weight: &extracted.payload.gate_weight,
            up_weight: &extracted.payload.up_weight,
            down_weight: &extracted.payload.down_weight,
            input: &extracted.payload.input,
        };
        let output = match probe_gated_gelu_ffn_f32(request) {
            Ok(output) => output,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_GGUF_LAYER_NO_GO error={err}");
                return match err {
                    MediaTekProbeError::UnsupportedPlatform => 1,
                    _ => 2,
                };
            }
        };
        let metrics = match Metrics::new(&extracted.payload.expected, &output.output) {
            Ok(metrics) => metrics,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_GGUF_LAYER_NO_GO error={err}");
                return 2;
            }
        };
        let finite_ok = metrics.finite_count == extracted.payload.output_size;
        let error_ok =
            metrics.max_abs_error <= args.tolerance || metrics.max_rel_error <= args.tolerance;
        let cosine_ok = metrics.cosine_similarity >= 0.999;
        let support_ok = output.supported_ops.iter().all(|(_, supported)| *supported);
        let accelerator_ok =
            output.chosen_device_type == rnb_backend_mediatek::MEDIATEK_NNAPI_DEVICE_ACCELERATOR;

        println!("mediatek_gemma_ffn_probe=gguf_layer");
        println!("gguf={}", gguf.display());
        println!("layer={}", extracted.metadata.layer);
        println!("input_row={}", args.input_row);
        println!("input_size={}", extracted.payload.input_size);
        println!("ffn_inner_size={}", extracted.payload.ffn_inner_size);
        println!("output_size={}", extracted.payload.output_size);
        println!("chosen_device.name={}", output.chosen_device_name);
        println!("chosen_device.type={}", output.chosen_device_type);
        println!(
            "chosen_device.feature_level={}",
            output.chosen_device_feature_level
        );
        println!("chosen_device.version={}", output.chosen_device_version);
        for tensor in &extracted.metadata.tensors {
            println!(
                "tensor.{} type={:?} shape={:?} row_start={} rows_selected={} cols_selected={} bytes_per_row={}",
                tensor.name,
                tensor.ggml_type,
                tensor.shape,
                tensor.row_start,
                tensor.rows_selected,
                tensor.cols_selected,
                tensor.bytes_per_row
            );
        }
        for (name, supported) in &output.supported_ops {
            println!("supported_ops.{name}={supported}");
        }
        println!("duration_hardware_ns={:?}", output.duration_hardware_ns);
        println!("duration_driver_ns={:?}", output.duration_driver_ns);
        println!("finite_count={}", metrics.finite_count);
        println!("output_len={}", output.output.len());
        println!("max_abs_error={:.9}", metrics.max_abs_error);
        println!("max_rel_error={:.9}", metrics.max_rel_error);
        println!("cosine_similarity={:.9}", metrics.cosine_similarity);

        if finite_ok && error_ok && cosine_ok && support_ok && accelerator_ok {
            println!("{SUCCESS_GGUF_LAYER}");
            0
        } else {
            eprintln!(
                "MTK_NNAPI_GEMMA_FFN_GGUF_LAYER_NO_GO finite_ok={finite_ok} \
                 error_ok={error_ok} cosine_ok={cosine_ok} support_ok={support_ok} \
                 accelerator_ok={accelerator_ok}"
            );
            1
        }
    }

    fn run_gguf_layer_batched(args: &Args, batch: usize) -> i32 {
        let Some(gguf) = args.gguf.as_ref() else {
            eprintln!("one mode is required: --synthetic or --gguf PATH");
            return 2;
        };
        if args.prompt_file.is_some() {
            eprintln!("--prompt-file is reserved for the token smoke path; batched probe uses --input-row");
            return 2;
        }
        eprintln!("[load] {}", gguf.display());
        let model = match rnb_loader::load_model(gguf) {
            Ok(model) => model,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_BATCHED_NO_GO load_model failed: {err:?}");
                return 1;
            }
        };
        let config = MtkGatedGeluFfnConfig {
            layer: args.layer.unwrap_or(0),
            input_row: args.input_row,
            input_scale: args.input_scale,
            ..MtkGatedGeluFfnConfig::default()
        };
        let extracted = match extract_gated_gelu_ffn_from_loaded_model(&model, &config) {
            Ok(extracted) => extracted,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_BATCHED_NO_GO extract failed: {err}");
                return 1;
            }
        };
        let input_size = extracted.payload.input_size;
        let output_size = extracted.payload.output_size;
        let mut batched_input = Vec::with_capacity(batch * input_size);
        for _ in 0..batch {
            batched_input.extend_from_slice(&extracted.payload.input);
        }
        let request = ProbeGatedGeluFfnF32BatchedRequest {
            device_name_substring: args.device.clone(),
            input_size,
            ffn_inner_size: extracted.payload.ffn_inner_size,
            output_size,
            gate_weight: &extracted.payload.gate_weight,
            up_weight: &extracted.payload.up_weight,
            down_weight: &extracted.payload.down_weight,
            input: &batched_input,
            batch,
            zero_copy: args.zerocopy,
            cache_dir: args.cache_dir.clone(),
        };
        let output = match probe_gated_gelu_ffn_f32_batched(request) {
            Ok(output) => output,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_BATCHED_NO_GO error={err}");
                return match err {
                    MediaTekProbeError::UnsupportedPlatform => 1,
                    _ => 2,
                };
            }
        };
        let mut reference = Vec::with_capacity(batch * output_size);
        for _ in 0..batch {
            reference.extend_from_slice(&extracted.payload.expected);
        }
        let metrics = match Metrics::new(&reference, &output.output) {
            Ok(metrics) => metrics,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_BATCHED_NO_GO parity error={err}");
                return 1;
            }
        };
        let npu_per_token_ns = output.execute_hw_ns.map(|hw| hw as f64 / batch as f64);
        let support_ok = output.supported;
        let finite_ok = metrics.finite_count == batch * output_size;
        let hw_ok = output.execute_hw_ns.is_some();
        let parity_ok = (metrics.max_abs_error <= 1.0e-4 || metrics.max_rel_error <= 1.0e-3)
            && metrics.cosine_similarity >= 0.999;

        println!("mediatek_gemma_ffn_probe=gguf_layer_batched");
        println!("gguf={}", gguf.display());
        println!("layer={}", extracted.metadata.layer);
        println!("batch={batch}");
        println!("zerocopy={}", args.zerocopy);
        println!("input_size={input_size}");
        println!("ffn_inner_size={}", extracted.payload.ffn_inner_size);
        println!("output_size={output_size}");
        println!("chosen_device.name={}", output.chosen_device_name);
        println!("chosen_device.type={}", output.chosen_device_type);
        for (name, supported) in &output.supported_ops {
            println!("supported_ops.{name}={supported}");
        }
        println!("supported={}", output.supported);
        println!("compile_ns={}", output.compile_ns);
        println!("token_hash_ns={}", output.token_hash_ns);
        println!("execute_hw_ns={:?}", output.execute_hw_ns);
        if let Some(npt) = npu_per_token_ns {
            println!("npu_per_token_ns={npt:.1}");
        }
        println!("execution_compute_ns={}", output.execution_compute_ns);
        println!(
            "npu_full_per_token_ns={:.1}",
            output.execution_compute_ns as f64 / batch as f64
        );
        println!("output_len={}", output.output.len());
        println!("finite_count={}", metrics.finite_count);
        println!("parity_max_abs_error={:.9}", metrics.max_abs_error);
        println!("parity_max_rel_error={:.9}", metrics.max_rel_error);
        println!("parity_cosine_similarity={:.9}", metrics.cosine_similarity);
        println!("output_checksum={}", f32_slice_checksum(&output.output));

        if support_ok && finite_ok && hw_ok && parity_ok {
            println!("MTK_NNAPI_GEMMA_FFN_BATCHED_OK");
            0
        } else {
            eprintln!(
                "MTK_NNAPI_GEMMA_FFN_BATCHED_NO_GO support_ok={support_ok} finite_ok={finite_ok} hw_ok={hw_ok} parity_ok={parity_ok}"
            );
            1
        }
    }

    fn run_gguf_layer_co_resident(args: &Args, requested_layers: usize) -> i32 {
        let Some(gguf) = args.gguf.as_ref() else {
            eprintln!("--co-resident requires --gguf PATH");
            return 2;
        };
        if args.prompt_file.is_some() {
            eprintln!("--prompt-file is reserved for the token smoke path; co-resident probe uses --input-row");
            return 2;
        }
        if args.layer.is_some() {
            eprintln!(
                "[co-resident] --layer is ignored; compiling layers 0..{}",
                requested_layers
            );
        }
        let batch = args.batch.unwrap_or(DEFAULT_CO_RESIDENT_BATCH);
        eprintln!("[load] {}", gguf.display());
        let model = match rnb_loader::load_model(gguf) {
            Ok(model) => model,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_CO_RESIDENT_NO_GO load_model failed: {err:?}");
                return 1;
            }
        };

        let rss_before_kb = read_vm_rss_kb_or_zero();
        let mem_available_before_kb = read_mem_available_kb_or_zero();
        let mut compilations: Vec<(usize, MediaTekGatedGeluFfnBatchedCompilation)> =
            Vec::with_capacity(requested_layers);
        let mut input_rows: Vec<Vec<f32>> = Vec::with_capacity(requested_layers);
        let mut expected_rows: Vec<Vec<f32>> = Vec::with_capacity(requested_layers);
        let mut compile_failed = false;

        for layer in 0..requested_layers {
            let config = MtkGatedGeluFfnConfig {
                layer,
                input_row: args.input_row,
                input_scale: args.input_scale,
                ..MtkGatedGeluFfnConfig::default()
            };
            let extracted = match extract_gated_gelu_ffn_from_loaded_model(&model, &config) {
                Ok(extracted) => extracted,
                Err(err) => {
                    eprintln!(
                        "MTK_NNAPI_GEMMA_FFN_CO_RESIDENT_NO_GO layer={layer} extract failed: {err}"
                    );
                    compile_failed = true;
                    break;
                }
            };
            let input_size = extracted.payload.input_size;
            let ffn_inner_size = extracted.payload.ffn_inner_size;
            let output_size = extracted.payload.output_size;
            let compilation = match if args.zerocopy || args.cache_dir.is_some() {
                compile_gated_gelu_ffn_f32_batched_with(
                    &args.device,
                    input_size,
                    ffn_inner_size,
                    output_size,
                    &extracted.payload.gate_weight,
                    &extracted.payload.up_weight,
                    &extracted.payload.down_weight,
                    batch,
                    BatchedCompileOptions {
                        zero_copy: args.zerocopy,
                        cache_dir: args.cache_dir.clone(),
                    },
                )
            } else {
                compile_gated_gelu_ffn_f32_batched(
                    &args.device,
                    input_size,
                    ffn_inner_size,
                    output_size,
                    &extracted.payload.gate_weight,
                    &extracted.payload.up_weight,
                    &extracted.payload.down_weight,
                    batch,
                )
            } {
                Ok(compilation) => compilation,
                Err(err) => {
                    eprintln!(
                        "MTK_NNAPI_GEMMA_FFN_CO_RESIDENT_NO_GO layer={layer} compile failed: {err}"
                    );
                    println!("co_resident_layer.{layer}.compile_ok=false");
                    compile_failed = true;
                    break;
                }
            };
            println!("co_resident_layer.{layer}.compile_ok=true");
            println!(
                "co_resident_layer.{layer}.compile_ns={}",
                compilation.timings().compilation_ns()
            );
            println!(
                "co_resident_layer.{layer}.token_hash_ns={}",
                compilation.timings().token_hash_ns()
            );
            input_rows.push(extracted.payload.input);
            expected_rows.push(extracted.payload.expected);
            compilations.push((layer, compilation));
        }

        let rss_after_kb = read_vm_rss_kb_or_zero();
        let mem_available_after_kb = read_mem_available_kb_or_zero();
        print_co_resident_memory_metrics(
            requested_layers,
            compilations.len(),
            batch,
            args.zerocopy,
            args.cache_dir.as_deref(),
            rss_before_kb,
            rss_after_kb,
            mem_available_before_kb,
            mem_available_after_kb,
        );

        if compile_failed || compilations.len() != requested_layers {
            println!("warm_run_ok=false");
            return 1;
        }

        let mut warm_run_ok = true;
        for ((layer, compilation), (input_row, expected_row)) in compilations
            .iter()
            .zip(input_rows.iter().zip(expected_rows.iter()))
        {
            let mut batched_input = Vec::with_capacity(batch * input_row.len());
            for _ in 0..batch {
                batched_input.extend_from_slice(input_row);
            }
            match run_compiled_gated_gelu_ffn_f32_batched(compilation, &batched_input, batch) {
                Ok(output) => {
                    let output_len_ok = output.output.len() == compilation.output_len();
                    println!(
                        "co_resident_layer.{layer}.execution_compute_ns={}",
                        output.execution_compute_ns
                    );
                    println!(
                        "co_resident_layer.{layer}.execute_hw_ns={:?}",
                        output.execute_hw_ns
                    );
                    println!(
                        "co_resident_layer.{layer}.output_checksum={}",
                        f32_slice_checksum(&output.output)
                    );
                    let mut parity_ok = false;
                    if output_len_ok {
                        let mut reference = Vec::with_capacity(batch * expected_row.len());
                        for _ in 0..batch {
                            reference.extend_from_slice(expected_row);
                        }
                        match Metrics::new(&reference, &output.output) {
                            Ok(metrics) => {
                                println!(
                                    "co_resident_layer.{layer}.parity_max_abs_error={:.9}",
                                    metrics.max_abs_error
                                );
                                println!(
                                    "co_resident_layer.{layer}.parity_max_rel_error={:.9}",
                                    metrics.max_rel_error
                                );
                                println!(
                                    "co_resident_layer.{layer}.parity_cosine_similarity={:.9}",
                                    metrics.cosine_similarity
                                );
                                parity_ok = (metrics.max_abs_error <= 1.0e-4
                                    || metrics.max_rel_error <= 1.0e-3)
                                    && metrics.cosine_similarity >= 0.999;
                            }
                            Err(err) => {
                                eprintln!(
                                    "MTK_NNAPI_GEMMA_FFN_CO_RESIDENT_NO_GO layer={layer} parity error={err}"
                                );
                            }
                        }
                    }
                    let layer_ok = output.batch == batch
                        && output.supported
                        && output_len_ok
                        && output.execute_hw_ns.is_some()
                        && parity_ok;
                    if !layer_ok {
                        eprintln!(
                            "MTK_NNAPI_GEMMA_FFN_CO_RESIDENT_NO_GO layer={layer} warm_run supported={} output_len_ok={} execute_hw_ns_present={} parity_ok={}",
                            output.supported,
                            output_len_ok,
                            output.execute_hw_ns.is_some(),
                            parity_ok
                        );
                        warm_run_ok = false;
                    }
                }
                Err(err) => {
                    eprintln!(
                        "MTK_NNAPI_GEMMA_FFN_CO_RESIDENT_NO_GO layer={layer} warm_run failed: {err}"
                    );
                    warm_run_ok = false;
                    break;
                }
            }
        }
        println!("warm_run_ok={warm_run_ok}");
        if warm_run_ok {
            println!("{SUCCESS_GGUF_CO_RESIDENT}");
            0
        } else {
            1
        }
    }

    fn f32_slice_checksum(values: &[f32]) -> u64 {
        // FNV-1a over native f32 byte patterns: deterministic per process so two
        // runs sharing the same compiled artifact + input produce the same checksum
        // (process-A vs process-B bit-parity for the AOT cache de-risk).
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in values {
            for byte in value.to_ne_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn print_co_resident_memory_metrics(
        requested_layers: usize,
        compiled_layers: usize,
        batch: usize,
        zerocopy: bool,
        cache_dir: Option<&str>,
        rss_before_kb: u64,
        rss_after_kb: u64,
        mem_available_before_kb: u64,
        mem_available_after_kb: u64,
    ) {
        let rss_delta_kb = rss_after_kb as i64 - rss_before_kb as i64;
        let mem_available_delta_kb = mem_available_after_kb as i64 - mem_available_before_kb as i64;
        let mem_available_consumed_kb =
            mem_available_before_kb as i64 - mem_available_after_kb as i64;
        let per_layer_kb = if compiled_layers > 0 {
            format!("{:.1}", rss_delta_kb as f64 / compiled_layers as f64)
        } else {
            "NA".to_string()
        };
        println!("co_resident_layers={requested_layers}");
        println!("co_resident_compiled_layers={compiled_layers}");
        println!("batch={batch}");
        println!("zerocopy={zerocopy}");
        println!("aot_cache_dir={}", cache_dir.unwrap_or("NA"));
        println!("rss_before_kb={rss_before_kb}");
        println!("rss_after_kb={rss_after_kb}");
        println!("rss_delta_kb={rss_delta_kb}");
        println!("per_layer_kb={per_layer_kb}");
        println!("mem_available_before_kb={mem_available_before_kb}");
        println!("mem_available_after_kb={mem_available_after_kb}");
        println!("mem_available_delta_kb={mem_available_delta_kb}");
        println!("mem_available_consumed_kb={mem_available_consumed_kb}");
    }

    fn read_vm_rss_kb_or_zero() -> u64 {
        match read_proc_kb_value("/proc/self/status", "VmRSS:") {
            Ok(value) => value,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_CO_RESIDENT_WARN VmRSS unavailable: {err}");
                0
            }
        }
    }

    fn read_mem_available_kb_or_zero() -> u64 {
        match read_proc_kb_value("/proc/meminfo", "MemAvailable:") {
            Ok(value) => value,
            Err(err) => {
                eprintln!("MTK_NNAPI_GEMMA_FFN_CO_RESIDENT_WARN MemAvailable unavailable: {err}");
                0
            }
        }
    }

    fn read_proc_kb_value(path: &str, key: &str) -> Result<u64, String> {
        let content =
            std::fs::read_to_string(path).map_err(|err| format!("read {path} failed: {err}"))?;
        for line in content.lines() {
            let Some(rest) = line.strip_prefix(key) else {
                continue;
            };
            let value = rest
                .split_whitespace()
                .next()
                .ok_or_else(|| format!("{key} value missing in {path}"))?;
            return value
                .parse::<u64>()
                .map_err(|err| format!("{key} value '{value}' in {path} is not u64: {err}"));
        }
        Err(format!("{key} not found in {path}"))
    }

    fn print_help() {
        println!(
            "Usage: rnb-mtk-gemma-ffn-probe --synthetic [--device TEXT] [--tolerance FLOAT]\n\
             \n\
                    rnb-mtk-gemma-ffn-probe --gguf PATH [--layer N] [--input-row N]\n\
             \n\
                    rnb-mtk-gemma-ffn-probe --gguf PATH --batch N [--zerocopy]\n\
             \n\
                    rnb-mtk-gemma-ffn-probe --gguf PATH --co-resident N [--batch N] [--zerocopy] [--cache-dir PATH]\n\
             \n\
             Options:\n\
               --synthetic              Run deterministic FLOAT32 gated GELU FFN probe\n\
               --gguf PATH              Run one GGUF-derived Gemma FFN layer through NNAPI\n\
               --co-resident N          Compile and hold layers 0..N, then warm-run each once\n\
               --batch N                Batched probe size; co-resident default 128\n\
               --zerocopy               Use f32 NNAPI shared-memory weight operands (GGUF modes only)\n\
               --cache-dir PATH         Enable NNAPI setCaching with PATH for co-resident compile\n\
               --layer N                GGUF layer index, default 0\n\
               --input-row N            token_embd row used as input, default 0\n\
               --input-scale FLOAT      finite input multiplier, default 1.0\n\
               --prompt-file PATH       Reserved for token smoke path\n\
               --device TEXT            NNAPI device name substring, default mtk-neuron\n\
               --tolerance FLOAT        max abs tolerance, default 0.001\n\
               --help                   Print this help"
        );
    }

    struct Args {
        synthetic: bool,
        gguf: Option<PathBuf>,
        layer: Option<usize>,
        input_row: usize,
        input_scale: f32,
        prompt_file: Option<String>,
        device: String,
        tolerance: f32,
        batch: Option<usize>,
        co_resident: Option<usize>,
        zerocopy: bool,
        cache_dir: Option<String>,
        help: bool,
    }

    impl Args {
        fn parse<I>(mut raw: I) -> Result<Self, String>
        where
            I: Iterator<Item = String>,
        {
            let mut args = Self {
                synthetic: false,
                gguf: None,
                layer: None,
                input_row: 0,
                input_scale: 1.0,
                prompt_file: None,
                device: "mtk-neuron".to_string(),
                tolerance: 0.001,
                batch: None,
                co_resident: None,
                zerocopy: false,
                cache_dir: None,
                help: false,
            };
            while let Some(arg) = raw.next() {
                match arg.as_str() {
                    "--help" | "-h" => args.help = true,
                    "--synthetic" => args.synthetic = true,
                    "--gguf" => args.gguf = Some(PathBuf::from(next_value(&mut raw, "--gguf")?)),
                    "--layer" => {
                        let value = next_value(&mut raw, "--layer")?;
                        args.layer = Some(
                            value
                                .parse::<usize>()
                                .map_err(|_| format!("invalid --layer value: {value}"))?,
                        );
                    }
                    "--input-row" => {
                        let value = next_value(&mut raw, "--input-row")?;
                        args.input_row = value
                            .parse::<usize>()
                            .map_err(|_| format!("invalid --input-row value: {value}"))?;
                    }
                    "--input-scale" => {
                        let value = next_value(&mut raw, "--input-scale")?;
                        args.input_scale = value
                            .parse::<f32>()
                            .map_err(|_| format!("invalid --input-scale value: {value}"))?;
                    }
                    "--prompt-file" => {
                        args.prompt_file = Some(next_value(&mut raw, "--prompt-file")?)
                    }
                    "--device" => args.device = next_value(&mut raw, "--device")?,
                    "--tolerance" => {
                        let value = next_value(&mut raw, "--tolerance")?;
                        args.tolerance = value
                            .parse::<f32>()
                            .map_err(|_| format!("invalid --tolerance value: {value}"))?;
                    }
                    "--batch" => {
                        let value = next_value(&mut raw, "--batch")?;
                        args.batch = Some(
                            value
                                .parse::<usize>()
                                .map_err(|_| format!("invalid --batch value: {value}"))?,
                        );
                    }
                    "--co-resident" => {
                        let value = next_value(&mut raw, "--co-resident")?;
                        args.co_resident = Some(
                            value
                                .parse::<usize>()
                                .map_err(|_| format!("invalid --co-resident value: {value}"))?,
                        );
                    }
                    "--zerocopy" => args.zerocopy = true,
                    "--cache-dir" => args.cache_dir = Some(next_value(&mut raw, "--cache-dir")?),
                    other => return Err(format!("unknown argument: {other}")),
                }
            }
            if args.device.trim().is_empty() {
                return Err("--device must not be empty".to_string());
            }
            if !args.tolerance.is_finite() || args.tolerance <= 0.0 {
                return Err("--tolerance must be finite and positive".to_string());
            }
            if matches!(args.batch, Some(0)) {
                return Err("--batch must be greater than zero".to_string());
            }
            if matches!(args.co_resident, Some(0)) {
                return Err("--co-resident must be greater than zero".to_string());
            }
            if args.co_resident.is_some() && args.gguf.is_none() {
                return Err("--co-resident requires --gguf PATH".to_string());
            }
            if args.zerocopy && args.gguf.is_none() {
                return Err("--zerocopy requires --gguf PATH".to_string());
            }
            if let Some(cache_dir) = args.cache_dir.as_deref() {
                if cache_dir.trim().is_empty() {
                    return Err("--cache-dir must not be empty".to_string());
                }
            }
            if args.synthetic && args.co_resident.is_some() {
                return Err("--co-resident cannot be combined with --synthetic".to_string());
            }
            if !args.input_scale.is_finite() {
                return Err("--input-scale must be finite".to_string());
            }
            if args.synthetic && args.gguf.is_some() {
                return Err("--synthetic cannot be combined with --gguf".to_string());
            }
            Ok(args)
        }
    }

    fn next_value<I>(raw: &mut I, flag: &'static str) -> Result<String, String>
    where
        I: Iterator<Item = String>,
    {
        raw.next()
            .filter(|value| !value.starts_with("--"))
            .ok_or_else(|| format!("{flag} requires a value"))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn parse(args: &[&str]) -> Result<Args, String> {
            Args::parse(args.iter().map(|arg| (*arg).to_string()))
        }

        #[test]
        fn args_default_to_copy_without_aot_cache() {
            let args = parse(&["--gguf", "model.gguf", "--batch", "2"]).expect("valid args");

            assert!(!args.zerocopy);
            assert_eq!(args.cache_dir, None);
        }

        #[test]
        fn args_parse_zerocopy_and_cache_dir() {
            let args = parse(&[
                "--gguf",
                "model.gguf",
                "--co-resident",
                "1",
                "--zerocopy",
                "--cache-dir",
                "/tmp/rnb-mk28-cache",
            ])
            .expect("valid zerocopy cache args");

            assert!(args.zerocopy);
            assert_eq!(args.cache_dir.as_deref(), Some("/tmp/rnb-mk28-cache"));
        }

        #[test]
        fn args_reject_empty_cache_dir() {
            let err = parse(&["--gguf", "model.gguf", "--cache-dir", " "])
                .expect_err("empty cache dir must be rejected");

            assert_eq!(err, "--cache-dir must not be empty");
        }

        #[test]
        fn args_reject_zerocopy_without_gguf() {
            let err = parse(&["--zerocopy"]).expect_err("zerocopy requires GGUF");

            assert_eq!(err, "--zerocopy requires --gguf PATH");
        }
    }

    struct SyntheticCase {
        input_size: usize,
        ffn_inner_size: usize,
        output_size: usize,
        input: Vec<f32>,
        gate_weight: Vec<f32>,
        up_weight: Vec<f32>,
        down_weight: Vec<f32>,
    }

    impl SyntheticCase {
        fn new() -> Self {
            Self {
                input_size: 4,
                ffn_inner_size: 3,
                output_size: 2,
                input: vec![0.25, -0.5, 0.75, 1.0],
                gate_weight: vec![
                    0.10, -0.20, 0.30, 0.40, -0.30, 0.20, 0.10, -0.10, 0.50, 0.25, -0.15, 0.05,
                ],
                up_weight: vec![
                    0.20, 0.10, -0.10, 0.30, -0.40, 0.25, 0.15, 0.05, 0.35, -0.20, 0.45, -0.25,
                ],
                down_weight: vec![0.30, -0.10, 0.25, -0.20, 0.40, 0.15],
            }
        }

        fn cpu_reference(&self) -> Vec<f32> {
            let gate = mat_vec(
                self.ffn_inner_size,
                self.input_size,
                &self.gate_weight,
                &self.input,
            );
            let up = mat_vec(
                self.ffn_inner_size,
                self.input_size,
                &self.up_weight,
                &self.input,
            );
            let gated = gate
                .iter()
                .zip(up.iter())
                .map(|(gate, up)| gelu(*gate) * up)
                .collect::<Vec<_>>();
            mat_vec(
                self.output_size,
                self.ffn_inner_size,
                &self.down_weight,
                &gated,
            )
        }
    }

    #[derive(Debug)]
    struct Metrics {
        finite_count: usize,
        max_abs_error: f32,
        max_rel_error: f32,
        cosine_similarity: f32,
    }

    impl Metrics {
        fn new(reference: &[f32], actual: &[f32]) -> Result<Self, String> {
            if reference.len() != actual.len() {
                return Err(format!(
                    "length mismatch: reference={} actual={}",
                    reference.len(),
                    actual.len()
                ));
            }
            let mut finite_count = 0usize;
            let mut max_abs_error = 0.0f32;
            let mut max_rel_error = 0.0f32;
            let mut dot = 0.0f64;
            let mut reference_norm = 0.0f64;
            let mut actual_norm = 0.0f64;
            for (idx, (reference, actual)) in reference.iter().zip(actual.iter()).enumerate() {
                if !reference.is_finite() || !actual.is_finite() {
                    return Err(format!("non-finite output at index {idx}"));
                }
                finite_count += 1;
                let abs_error = (reference - actual).abs();
                max_abs_error = max_abs_error.max(abs_error);
                let denom = reference.abs().max(1e-12);
                max_rel_error = max_rel_error.max(abs_error / denom);
                dot += f64::from(*reference) * f64::from(*actual);
                reference_norm += f64::from(*reference) * f64::from(*reference);
                actual_norm += f64::from(*actual) * f64::from(*actual);
            }
            let cosine_similarity = if reference_norm > 0.0 && actual_norm > 0.0 {
                (dot / (reference_norm.sqrt() * actual_norm.sqrt())) as f32
            } else if reference_norm == actual_norm {
                1.0
            } else {
                0.0
            };
            Ok(Self {
                finite_count,
                max_abs_error,
                max_rel_error,
                cosine_similarity,
            })
        }
    }

    fn mat_vec(rows: usize, cols: usize, weight: &[f32], input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            let mut acc = 0.0f32;
            let offset = row * cols;
            for col in 0..cols {
                acc += weight[offset + col] * input[col];
            }
            output[row] = acc;
        }
        output
    }

    fn gelu(value: f32) -> f32 {
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        0.5 * value * (1.0 + (sqrt_2_over_pi * (value + 0.044715 * value.powi(3))).tanh())
    }
}
