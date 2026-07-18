fn parse_affinity_sweep_specs(raw: &str) -> Vec<String> {
    raw.split(';')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect()
}

fn effective_affinity_label(explicit: Option<&str>, legacy_big_cores: bool) -> String {
    explicit
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| {
            if legacy_big_cores {
                "big(legacy)".to_string()
            } else if cfg!(target_arch = "aarch64") {
                "auto(default)".to_string()
            } else {
                "all(default)".to_string()
            }
        })
}

fn cpu_freq_trace_enabled() -> bool {
    std::env::var("RNB_CPU_FREQ_TRACE").is_ok()
}

fn llm_bench_help_requested<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    args.into_iter()
        .any(|arg| matches!(arg.as_ref(), "-h" | "--help"))
}

fn llm_bench_usage() -> &'static str {
    "usage: rnb-llm-bench\n\
     \n\
     This development benchmark is configured through environment variables.\n\
     Common variables:\n\
       RNB_MODEL=<gguf path>\n\
       RNB_FORCE_GGUF=1\n\
       RNB_PROMPT_FILE=<prompt text path>\n\
       RNB_PREFILL_TOKENS=<token count>\n\
       RNB_DECODE_TOKENS=<token count>\n\
       RNB_BENCH_WALL=1\n\
       RNB_QUIET_DECODE=1"
}

const HOST_RAM_BUDGET_ENV: &str = "RNB_HOST_RAM_BUDGET_BYTES";

fn parse_host_ram_budget_bytes(raw: Option<&str>) -> Result<Option<u64>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "{HOST_RAM_BUDGET_ENV} must be decimal u64 bytes, got {raw:?}"
        ));
    }
    raw.parse::<u64>()
        .map(Some)
        .map_err(|_| format!("{HOST_RAM_BUDGET_ENV} must be decimal u64 bytes, got {raw:?}"))
}

fn host_ram_budget_bytes_from_env() -> Result<Option<u64>, String> {
    match std::env::var(HOST_RAM_BUDGET_ENV) {
        Ok(raw) => parse_host_ram_budget_bytes(Some(&raw)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{HOST_RAM_BUDGET_ENV} must be decimal u64 bytes"))
        }
    }
}

#[cfg(feature = "mediatek")]
fn mediatek_quant_gated_gelu_probe_requested() -> bool {
    std::env::var("RNB_MEDIATEK_QUANT_GATED_GELU_SUPPORT_PROBE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !value.is_empty() && !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

#[cfg(feature = "mediatek")]
fn parse_mtk_quant_gated_gelu_probe_shape(
    raw: Option<&str>,
) -> Result<rnb_backend_mediatek::MediaTekGatedGeluFfnShape, String> {
    let raw = raw.unwrap_or("1536,6144,1536");
    let mut parts = raw.split(',').map(str::trim);
    let input_size = parse_mtk_quant_gated_gelu_dim(parts.next(), "input_size")?;
    let ffn_inner_size = parse_mtk_quant_gated_gelu_dim(parts.next(), "ffn_inner_size")?;
    let output_size = parse_mtk_quant_gated_gelu_dim(parts.next(), "output_size")?;
    if parts.next().is_some() {
        return Err(format!(
            "RNB_MEDIATEK_QUANT_GATED_GELU_SHAPE must be input,inner,output; got {raw}"
        ));
    }
    Ok(rnb_backend_mediatek::MediaTekGatedGeluFfnShape::new(
        input_size,
        ffn_inner_size,
        output_size,
    ))
}

#[cfg(feature = "mediatek")]
fn parse_mtk_quant_gated_gelu_dim(raw: Option<&str>, name: &'static str) -> Result<usize, String> {
    let raw = raw.ok_or_else(|| {
        "RNB_MEDIATEK_QUANT_GATED_GELU_SHAPE must be input,inner,output".to_string()
    })?;
    let value = raw
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a positive integer; got {raw}"))?;
    if value == 0 {
        return Err(format!("{name} must be non-zero"));
    }
    Ok(value)
}

#[cfg(feature = "mediatek")]
fn format_mtk_quant_gated_gelu_probe_result(
    shape: rnb_backend_mediatek::MediaTekGatedGeluFfnShape,
    result: &rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnSupportResult,
) -> String {
    let mut supported_ops = String::new();
    for (idx, (name, supported)) in result
        .supported_ops()
        .named()
        .iter()
        .filter(|(name, _)| !matches!(*name, "gate_dequant" | "up_dequant" | "gated_quantize"))
        .enumerate()
    {
        if idx > 0 {
            supported_ops.push(',');
        }
        supported_ops.push_str(name);
        supported_ops.push('=');
        supported_ops.push_str(if *supported { "true" } else { "false" });
    }
    format!(
        "[mediatek-quant-ffn-probe] shape={}x{}x{} device={} device_type={} feature_level={} supported={} supported_ops={} model_build_ns={} supported_ops_query_ns={}",
        shape.input_size(),
        shape.ffn_inner_size(),
        shape.output_size(),
        result.chosen_device().name(),
        result.chosen_device().device_type(),
        result.chosen_device().feature_level(),
        result.supported(),
        supported_ops,
        result.model_build_ns(),
        result.supported_ops_query_ns(),
    )
}

#[cfg(feature = "mediatek")]
fn run_mtk_quant_gated_gelu_probe_from_env() -> Result<(), String> {
    let shape = parse_mtk_quant_gated_gelu_probe_shape(
        std::env::var("RNB_MEDIATEK_QUANT_GATED_GELU_SHAPE")
            .ok()
            .as_deref(),
    )?;
    let device = std::env::var("RNB_MEDIATEK_DEVICE").unwrap_or_else(|_| "mtk-neuron".to_string());
    let probe = rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnSupportProbe::new(shape)
        .with_device_name_substring(device);
    let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
    let result = backend
        .probe_quantized_gated_gelu_ffn_support(&probe)
        .map_err(|err| err.to_string())?;
    println!(
        "{}",
        format_mtk_quant_gated_gelu_probe_result(shape, &result)
    );
    Ok(())
}

fn generated_text_sha256_hex(text: &str) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn dump_generated_text_hash_if_requested(result: &rnb_llm::generate::GenerateResult) {
    if std::env::var("RNB_DUMP_GENERATED_TEXT_HASH").is_err() {
        return;
    }
    eprintln!(
        "RNB_GENERATED_TEXT_HASH sha256={} bytes={} chars={} tokens={}",
        generated_text_sha256_hex(&result.text),
        result.text.len(),
        result.text.chars().count(),
        result.tokens_generated
    );
}

fn cpu_freq_snapshot() -> String {
    let mut parts = Vec::new();
    for cpu in 4..=7 {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq");
        let freq = std::fs::read_to_string(path)
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "na".to_string());
        parts.push(format!("cpu{cpu}={freq}"));
    }
    parts.join(" ")
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn format_vulkan_runtime_counters(
    submits: u64,
    upload_bytes: u64,
    download_bytes: u64,
    materializations: u64,
    attention_fan_in_copies: u64,
) -> String {
    format!(
        "[vulkan:counters] submits={submits} upload_bytes={upload_bytes} \
         download_bytes={download_bytes} materializations={materializations} \
         attention_fan_in_copies={attention_fan_in_copies}"
    )
}

fn dump_top_logits(engine: &rnb_llm::Engine, logits: &[f32], top_k: usize, label: &str) {
    let mut pairs = Vec::with_capacity(logits.len());
    let mut nan_count = 0usize;
    let mut pos_inf_count = 0usize;
    let mut neg_inf_count = 0usize;
    for (token_id, logit) in logits.iter().copied().enumerate() {
        if logit.is_finite() {
            pairs.push((token_id, logit));
        } else if logit.is_nan() {
            nan_count += 1;
        } else if logit.is_sign_positive() {
            pos_inf_count += 1;
        } else {
            neg_inf_count += 1;
        }
    }
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    eprintln!("=== top logits: {label} ===");
    match pairs.as_slice() {
        [(top1_id, top1), (top2_id, top2), rest @ ..] => {
            let margin13 = rest
                .first()
                .map(|(_, top3)| top1 - top3)
                .unwrap_or(top1 - top2);
            let top1_piece = engine
                .tokenizer
                .decode_token(*top1_id as u32)
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            let top2_piece = engine
                .tokenizer
                .decode_token(*top2_id as u32)
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            eprintln!(
                "[top-margin] label={label} finite={} nan={} pos_inf={} neg_inf={} \
                 top1_id={} top2_id={} margin12={:.6} margin13={:.6} \
                 top1={:.6} top2={:.6} near_tie={} top1_piece={:?} top2_piece={:?}",
                pairs.len(),
                nan_count,
                pos_inf_count,
                neg_inf_count,
                top1_id,
                top2_id,
                top1 - top2,
                margin13,
                top1,
                top2,
                (top1 - top2).abs() <= 0.5,
                top1_piece,
                top2_piece,
            );
        }
        [(top1_id, top1)] => {
            let top1_piece = engine
                .tokenizer
                .decode_token(*top1_id as u32)
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            eprintln!(
                "[top-margin] label={label} finite={} nan={} pos_inf={} neg_inf={} \
                 top1_id={} top1={:.6} top1_piece={:?}",
                pairs.len(),
                nan_count,
                pos_inf_count,
                neg_inf_count,
                top1_id,
                top1,
                top1_piece,
            );
        }
        [] => {
            eprintln!(
                "[top-margin] label={label} finite=0 nan={} pos_inf={} neg_inf={}",
                nan_count, pos_inf_count, neg_inf_count
            );
        }
    }
    for (rank, (token_id, logit)) in pairs.into_iter().take(top_k).enumerate() {
        let piece = engine
            .tokenizer
            .decode_token(token_id as u32)
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        eprintln!(
            "#{:02} id={} logit={:.6} piece={}",
            rank + 1,
            token_id,
            logit,
            piece
        );
    }
}

fn dump_tokens(engine: &rnb_llm::Engine, tokens: &[u32], label: &str) {
    eprintln!("=== tokens: {label} (n={}) ===", tokens.len());
    for (idx, &token) in tokens.iter().enumerate() {
        let piece = engine.tokenizer.decode_token(token).replace('\n', "\\n");
        eprintln!("#{idx:02} id={} piece={}", token, piece);
    }
}

fn dump_target_ranks(engine: &rnb_llm::Engine, logits: &[f32], pieces: &[&str], label: &str) {
    if pieces.is_empty() {
        return;
    }
    let mut pairs = logits.iter().copied().enumerate().collect::<Vec<_>>();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("=== target ranks: {label} ===");
    for piece in pieces {
        let mut best: Option<(usize, usize, f32)> = None;
        for (rank_idx, (token_id, logit)) in pairs.iter().enumerate() {
            let tok_piece = engine.tokenizer.decode_token(*token_id as u32);
            if tok_piece == *piece {
                best = Some((rank_idx + 1, *token_id, *logit));
                break;
            }
        }
        match best {
            Some((rank, token_id, logit)) => {
                eprintln!("#{rank} id={token_id} logit={logit:.6} piece={:?}", piece);
            }
            None => {
                eprintln!("not found piece={:?}", piece);
            }
        }
    }
}

fn format_profile_reports(gemv: Option<String>, moe: Option<String>) -> Option<String> {
    let mut reports = Vec::new();
    if let Some(report) = gemv {
        reports.push(report);
    }
    if let Some(report) = moe {
        reports.push(report);
    }
    (!reports.is_empty()).then(|| reports.join("\n"))
}

fn print_profile_reports() {
    if let Some(report) = format_profile_reports(
        rnb_llm::engine::gemv_profile_report(),
        rnb_llm::engine::moe_profile_report(),
    ) {
        eprintln!("\n{}", report);
    }
    if let Some(report) = rnb_llm::engine::packed_dispatch_report() {
        eprintln!("\n{}", report);
    }
}

fn parse_target_pieces_env() -> Vec<String> {
    std::env::var("RNB_DUMP_TARGET_PIECES")
        .ok()
        .map(|raw| {
            raw.split(';')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn suppress_special_logits(engine: &rnb_llm::Engine, logits: &mut [f32], force: bool) {
    if !force && std::env::var("RNB_SUPPRESS_SPECIAL_TOKENS").is_err() {
        return;
    }
    for (id, logit) in logits.iter_mut().enumerate() {
        let piece = engine.tokenizer.decode_token(id as u32);
        let looks_special = (piece.starts_with('<') && piece.ends_with('>'))
            || piece == "</s>"
            || piece == "<turn|>"
            || piece == "<|turn>"
            || piece == "<|channel>"
            || piece == "<channel|>"
            || piece == "<|think|>";
        if looks_special {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn suppress_selected_pieces(engine: &rnb_llm::Engine, logits: &mut [f32]) {
    let Ok(raw) = std::env::var("RNB_SUPPRESS_PIECES") else {
        return;
    };
    let targets = raw.split(';').filter(|s| !s.is_empty()).collect::<Vec<_>>();
    if targets.is_empty() {
        return;
    }
    for (id, logit) in logits.iter_mut().enumerate() {
        let piece = engine.tokenizer.decode_token(id as u32);
        if targets.iter().any(|target| piece == *target) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn run_prefill(engine: &mut rnb_llm::Engine, tokens: &[u32]) -> Vec<f32> {
    if std::env::var("RNB_TOKENWISE_PROMPT").is_ok() {
        let mut logits = vec![0.0f32; engine.metadata.vocab_size];
        for &token in tokens {
            logits = engine.forward(&[token]).unwrap();
        }
        logits
    } else {
        engine.forward(tokens).unwrap()
    }
}

/// argmax 토큰 인덱스 + 그 logit 값 반환 (reset 정확성 검증용).
fn argmax_logit(logits: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillAbabMode {
    Step5Gemm,
    Step45Gemm,
    FlashAttn,
    AttnChain,
    GdnFullFfn,
    AtnFullLayer,
    AtnOTail,
    MediatekFfn,
}

const PREFILL_ABAB_MODE_FLAGS: &[(PrefillAbabMode, &str)] = &[
    (PrefillAbabMode::Step5Gemm, "RNB_PREFILL_ABAB_STEP5GEMM"),
    (PrefillAbabMode::Step45Gemm, "RNB_PREFILL_ABAB_STEP45GEMM"),
    (PrefillAbabMode::FlashAttn, "RNB_PREFILL_ABAB_FLASH_ATTN"),
    (PrefillAbabMode::AttnChain, "RNB_PREFILL_ABAB_ATTN_CHAIN"),
    (PrefillAbabMode::GdnFullFfn, "RNB_PREFILL_ABAB_GDN_FULL_FFN"),
    (
        PrefillAbabMode::AtnFullLayer,
        "RNB_PREFILL_ABAB_ATN_FULL_LAYER",
    ),
    (PrefillAbabMode::AtnOTail, "RNB_PREFILL_ABAB_ATN_O_TAIL"),
    (
        PrefillAbabMode::MediatekFfn,
        "RNB_PREFILL_ABAB_MEDIATEK_FFN",
    ),
];

fn prefill_abab_mode_from_lookup(
    mut lookup: impl FnMut(&str) -> Option<String>,
) -> Result<Option<PrefillAbabMode>, String> {
    let active: Vec<_> = PREFILL_ABAB_MODE_FLAGS
        .iter()
        .filter_map(|&(mode, var)| (lookup(var).as_deref() == Some("1")).then_some((mode, var)))
        .collect();
    match active.as_slice() {
        [] => Ok(None),
        [(mode, _)] => Ok(Some(*mode)),
        _ => Err(format!(
            "ABAB mode env 는 하나만 켜야 함: {}",
            active
                .iter()
                .map(|(_, var)| *var)
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn prefill_abab_mode_from_env() -> Result<Option<PrefillAbabMode>, String> {
    prefill_abab_mode_from_lookup(|var| std::env::var(var).ok())
}

fn prefill_abab_labels(mode: Option<PrefillAbabMode>) -> (&'static str, &'static str) {
    match mode {
        Some(PrefillAbabMode::GdnFullFfn) => ("gdn full+ffn", "gdn full + host ffn"),
        Some(PrefillAbabMode::AttnChain) => ("attn device chain", "f16 NEON CPU attn"),
        Some(PrefillAbabMode::FlashAttn) => ("flash attn GPU", "f16 NEON CPU attn"),
        Some(PrefillAbabMode::AtnFullLayer) => ("atn full/core ON", "atn full/core OFF"),
        Some(PrefillAbabMode::AtnOTail) => ("atn o-tail ON", "atn o-tail OFF"),
        Some(PrefillAbabMode::MediatekFfn) => ("mediatek prefill FFN", "CPU prefill FFN"),
        Some(PrefillAbabMode::Step5Gemm) | Some(PrefillAbabMode::Step45Gemm) | None => {
            ("chain ON", "chain OFF")
        }
    }
}

fn prefill_abab_label_scoped_argmax(mode: Option<PrefillAbabMode>) -> bool {
    matches!(
        mode,
        Some(PrefillAbabMode::FlashAttn) | Some(PrefillAbabMode::AttnChain)
    )
}

fn prefill_abab_reports_atn_counters(mode: Option<PrefillAbabMode>) -> bool {
    matches!(
        mode,
        Some(PrefillAbabMode::AtnFullLayer) | Some(PrefillAbabMode::AtnOTail)
    )
}

fn reset_prefill_abab_atn_counters(mode: Option<PrefillAbabMode>) {
    match mode {
        Some(PrefillAbabMode::AtnFullLayer) => rnb_llm::reset_metal_prefill_atn_full_counters(),
        Some(PrefillAbabMode::AtnOTail) => rnb_llm::reset_metal_prefill_atn_o_tail_counters(),
        _ => {}
    }
}

fn report_prefill_abab_atn_counters(mode: Option<PrefillAbabMode>, label: &str) {
    match mode {
        Some(PrefillAbabMode::AtnFullLayer) => {
            rnb_llm::report_metal_prefill_atn_full_counters(label)
        }
        Some(PrefillAbabMode::AtnOTail) => rnb_llm::report_metal_prefill_atn_o_tail_counters(label),
        _ => {}
    }
}

fn prefill_abab_env_updates(
    mode: Option<PrefillAbabMode>,
    enabled: bool,
) -> Vec<(&'static str, &'static str)> {
    match mode {
        Some(PrefillAbabMode::GdnFullFfn) => vec![
            ("RNB_METAL_PREFILL_GDN_FULL", "1"),
            (
                "RNB_METAL_PREFILL_GDN_FULL_FFN",
                if enabled { "1" } else { "0" },
            ),
        ],
        Some(PrefillAbabMode::AttnChain) => vec![(
            "RNB_METAL_PREFILL_ATTN_CHAIN",
            if enabled { "1" } else { "0" },
        )],
        Some(PrefillAbabMode::FlashAttn) => vec![(
            "RNB_METAL_PREFILL_FLASH_ATTN",
            if enabled { "1" } else { "0" },
        )],
        Some(PrefillAbabMode::Step45Gemm) => vec![
            ("RNB_METAL_PREFILL_GDN_FULL", "1"),
            ("RNB_METAL_PREFILL_GDN_CONV_DELTA", "1"),
            ("RNB_METAL_PREFILL_GATED_PROJ", "1"),
            (
                "RNB_METAL_PREFILL_DELTA_STEP45_GEMM",
                if enabled { "1" } else { "0" },
            ),
            (
                "RNB_METAL_PREFILL_DELTA_STEP5_GEMM",
                if enabled { "0" } else { "1" },
            ),
        ],
        Some(PrefillAbabMode::Step5Gemm) => vec![
            ("RNB_METAL_PREFILL_GDN_FULL", "1"),
            ("RNB_METAL_PREFILL_GDN_CONV_DELTA", "1"),
            ("RNB_METAL_PREFILL_GATED_PROJ", "1"),
            (
                "RNB_METAL_PREFILL_DELTA_STEP5_GEMM",
                if enabled { "1" } else { "0" },
            ),
        ],
        Some(PrefillAbabMode::AtnFullLayer) => vec![
            (
                "RNB_METAL_PREFILL_ATN_FULL_LAYER",
                if enabled { "1" } else { "0" },
            ),
            ("RNB_METAL_PREFILL_ATN_FULL_TIME", "1"),
        ],
        Some(PrefillAbabMode::AtnOTail) => vec![
            (
                "RNB_METAL_PREFILL_ATN_O_TAIL",
                if enabled { "1" } else { "0" },
            ),
            ("RNB_METAL_PREFILL_ATN_O_TAIL_TIME", "1"),
        ],
        Some(PrefillAbabMode::MediatekFfn) => {
            vec![("RNB_MEDIATEK_PREFILL_FFN", if enabled { "1" } else { "0" })]
        }
        None => {
            let val = if enabled { "1" } else { "0" };
            vec![
                ("RNB_METAL_PREFILL_GDN_FULL", val),
                ("RNB_METAL_PREFILL_GDN_CONV_DELTA", val),
                ("RNB_METAL_PREFILL_GATED_PROJ", val),
            ]
        }
    }
}

fn apply_prefill_abab_env(mode: Option<PrefillAbabMode>, enabled: bool) {
    // SAFETY: 단일 스레드 측정 하니스. 다른 스레드가 동시에 env 를 읽지 않는다.
    unsafe {
        for (var, val) in prefill_abab_env_updates(mode, enabled) {
            std::env::set_var(var, val);
        }
    }
}

/// 단일 프로세스 prefill ABAB 반복 측정 하니스 (D축 GDN prefill chain wall 측정용).
///
/// 동기: "매 run 별도 프로세스 + generate_ms" 측정은 매번 모델 재로딩 + cold start +
/// 프로세스 간 thermal 변동으로 run-to-run noise 가 ±300ms 라, 재려는 chain 효과
/// (수십~수백ms) 보다 noise 가 커서 판정 불가. 단일 프로세스 안에서 모델 1회 로드 후
/// 같은 prefill 을 ABAB 로 반복하면 cold start / 프로세스 간 변동이 사라져 wall 분산이
/// 확 줄어 chain 의 진짜 wall 효과를 본다.
///
/// A = chain full ON, B = chain 전부 OFF.
/// `RNB_METAL_PREFILL_GDN_FULL` / `RNB_METAL_PREFILL_GDN_CONV_DELTA` /
/// `RNB_METAL_PREFILL_GATED_PROJ` 는 metal seam(crate rnb-runtime metal_inference.rs)
/// 에서 prefill 호출마다 `std::env::var` 로 읽히므로 `set_var` 로 런타임 토글 가능.
///
/// 매 prefill 전 `engine.clear_sequence_state()` 로 KV cache(current_len=0) + SSM
/// conv/delta state(0 초기화) + metal backend carrier(`clear_sequence_state`) 를
/// 모두 reset → 매 prefill 이 첫 prefill 과 동일한 fresh 조건. reset 정확성은
/// 매 prefill 의 argmax/첫 logit 이 1회차와 동일한지로 검증한다.
fn run_prefill_abab(engine: &mut rnb_llm::Engine, tokens: &[u32], repeat: usize) {
    let mode = match prefill_abab_mode_from_env() {
        Ok(mode) => mode,
        Err(err) => {
            eprintln!("[ABAB] {err}");
            return;
        }
    };
    let mediatek_ffn_mode = matches!(mode, Some(PrefillAbabMode::MediatekFfn));
    let mediatek_no_prewarm =
        std::env::var("RNB_PREFILL_ABAB_MEDIATEK_FFN_NO_PREWARM").as_deref() == Ok("1");
    // chain A/B env 토글 (매 prefill 호출마다 metal seam 이 읽음).
    let set_chain = move |enabled: bool| {
        apply_prefill_abab_env(mode, enabled);
    };

    let reset = |engine: &mut rnb_llm::Engine| {
        engine
            .clear_sequence_state()
            .expect("clear_sequence_state failed in ABAB harness");
    };

    let (a_label, b_label) = prefill_abab_labels(mode);
    eprintln!(
        "[ABAB] prefill ABAB harness: prompt_tokens={}, repeat=N={} (= {} measured prefills), A={a_label} / B={b_label}",
        tokens.len(),
        repeat,
        2 * repeat
    );
    if mediatek_ffn_mode && !mediatek_no_prewarm {
        set_chain(true);
        #[cfg(feature = "mediatek")]
        {
            match engine.prewarm_mediatek_prefill_ffn(
                tokens.len(),
                rnb_runtime::mediatek::MediaTekPrefillRequestMode::BenchWarmup,
            ) {
                Ok(used_layers) => eprintln!(
                    "[ABAB] mediatek prewarm: request_mode=BenchWarmup used_layers={used_layers}"
                ),
                Err(err) => eprintln!("[ABAB] mediatek prewarm failed: {err}"),
            }
        }
        #[cfg(not(feature = "mediatek"))]
        {
            eprintln!(
                "[ABAB] mediatek prewarm requested but rnb-dev-tools was built without feature=mediatek"
            );
        }
    }

    // --- warmup (측정 제외, pipeline/carrier 캐시 워밍) ---
    let warmup_runs = if mediatek_ffn_mode && mediatek_no_prewarm {
        0
    } else {
        3usize
    };
    let mut ref_argmax: Option<(usize, f32)> = None;
    for w in 0..warmup_runs {
        // warmup 은 A(chain ON) 조건으로 — 측정 첫 A 와 동일 조건이라 carrier 워밍이 유효.
        set_chain(true);
        reset(engine);
        let logits = run_prefill(engine, tokens);
        let am = argmax_logit(&logits);
        if w == 0 {
            ref_argmax = Some(am);
        }
        eprintln!(
            "[ABAB] warmup {}/{}: argmax_token={} logit={:.6}",
            w + 1,
            warmup_runs,
            am.0,
            am.1
        );
    }
    let ref_argmax_token = ref_argmax.map(|am| am.0);

    // --- ABAB 2N 측정 ---
    let mut a_ms: Vec<f64> = Vec::with_capacity(repeat);
    let mut b_ms: Vec<f64> = Vec::with_capacity(repeat);
    let mut reset_ok = true;
    // pm48: flash/attn-chain modes 는 A(GPU f32 acc) vs B(CPU f16) 가 f16 drift 로
    // argmax 가 다를 수 있어 cross-condition 비교는 부적절. label 별 첫 argmax 를
    // reference 로 잡아 같은 조건 안에서만 reset 일관성을 검증한다. ATN full/core 는
    // token-identical 후보라 A/B 모두 단일 ref 를 따라야 한다.
    let mut a_ref: Option<usize> = None;
    let mut b_ref: Option<usize> = None;
    for i in 0..repeat {
        for &(label, chain_on) in &[("A", true), ("B", false)] {
            set_chain(chain_on);
            if prefill_abab_reports_atn_counters(mode) {
                reset_prefill_abab_atn_counters(mode);
            }
            reset(engine);
            let t = std::time::Instant::now();
            let logits = run_prefill(engine, tokens);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            if prefill_abab_reports_atn_counters(mode) {
                report_prefill_abab_atn_counters(mode, label);
            }
            let am = argmax_logit(&logits);
            // reset 정확성 검증: 매 prefill 의 argmax 가 동일 조건의 reference 와 같아야 함.
            let label_local_refs = prefill_abab_label_scoped_argmax(mode)
                || (mediatek_ffn_mode && mediatek_no_prewarm);
            let match_ref = if label_local_refs {
                // device chain/flash/prewarm-disabled MediaTek runs can differ between A and B,
                // so validate reset consistency within each label.
                let r = if chain_on { &mut a_ref } else { &mut b_ref };
                match *r {
                    Some(rv) => am.0 == rv,
                    None => {
                        *r = Some(am.0);
                        true
                    }
                }
            } else if let Some(rv) = ref_argmax_token {
                // chain ON/OFF 는 token-identical 설계라 A/B 모두 같은 argmax 여야 함.
                am.0 == rv
            } else {
                false
            };
            if !match_ref {
                reset_ok = false;
            }
            eprintln!(
                "[ABAB] iter {}/{} {}: wall={:.3}ms argmax_token={} logit={:.6} state_reset_ok={}",
                i + 1,
                repeat,
                label,
                ms,
                am.0,
                am.1,
                match_ref
            );
            if chain_on {
                a_ms.push(ms);
            } else {
                b_ms.push(ms);
            }
        }
    }

    // A1 (첫 A) 제외하고 median 비교.
    let median = |v: &[f64]| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        }
    };

    let a_excl_first: Vec<f64> = a_ms.iter().skip(1).copied().collect();
    let a_med = median(&a_excl_first);
    let b_med = median(&b_ms);
    let diff_ms = a_med - b_med;
    let diff_pct = if b_med != 0.0 {
        diff_ms / b_med * 100.0
    } else {
        f64::NAN
    };

    eprintln!("[ABAB] ---- summary ----");
    eprintln!(
        "[ABAB] A ({a_label}) raw ms: {:?}  (A1 excluded for median)",
        a_ms
    );
    eprintln!("[ABAB] B ({b_label}) raw ms: {:?}", b_ms);
    eprintln!(
        "[ABAB] A median (excl A1) = {:.3}ms over {} runs",
        a_med,
        a_excl_first.len()
    );
    eprintln!(
        "[ABAB] B median          = {:.3}ms over {} runs",
        b_med,
        b_ms.len()
    );
    eprintln!(
        "[ABAB] diff (A - B) = {:.3}ms ({:+.2}%)  [음수면 A({a_label}) 가 빠름]",
        diff_ms, diff_pct
    );
    if prefill_abab_label_scoped_argmax(mode) || (mediatek_ffn_mode && mediatek_no_prewarm) {
        eprintln!(
            "[ABAB] state_reset_correctness = {} (label 별 argmax 일관: A ref={:?} / B ref={:?})",
            if reset_ok { "OK" } else { "FAILED" },
            a_ref,
            b_ref
        );
    } else {
        eprintln!(
            "[ABAB] state_reset_correctness = {} (모든 prefill argmax == warmup argmax token {:?})",
            if reset_ok { "OK" } else { "FAILED" },
            ref_argmax_token
        );
    }
    if !reset_ok {
        eprintln!(
            "[ABAB] WARNING: reset 가 불완전 — 2번째 이후 prefill 이 fresh 조건이 아님. 측정 무효."
        );
    }
}

/// Dump (normed backbone_hidden, argmax_token, prev_token) per decode position
/// for drafter calibration (Stage C-2).
///
/// 포맷:
/// ```text
/// magic       : "RNBHTRC1" (8 bytes)
/// hidden_dim  : u32 (little-endian)
/// n_positions : u32 (little-endian)
/// for each position (decode step):
///   hidden    : f32 × hidden_dim (little-endian, row-major)
///   argmax    : u32  (target argmax token at this step)
///   prev_token: u32  (token fed into the forward that produced this hidden;
///                     == argmax of previous step, == last prompt token at step 0)
/// ```
fn dump_backbone_hidden_trace(
    engine: &mut rnb_llm::Engine,
    prompt_tokens: &[u32],
    decode_count: usize,
    dump_path: &str,
) {
    use std::io::Write;

    assert!(!prompt_tokens.is_empty(), "empty prompt for trace dump");
    eprintln!(
        "[TRACE_DUMP] prompt_tokens={}, decode_count={}, path={}",
        prompt_tokens.len(),
        decode_count,
        dump_path
    );

    // 1. Prefill the prompt — establishes KV cache up through last prompt token.
    let prefill_logits = run_prefill(engine, prompt_tokens);
    let first_token = rnb_llm::sampler::greedy::greedy_sample(&prefill_logits);
    let hidden_dim = engine.metadata.hidden_dim;

    let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(decode_count);
    let mut argmax_tokens: Vec<u32> = Vec::with_capacity(decode_count);
    let mut prev_tokens: Vec<u32> = Vec::with_capacity(decode_count);

    // Step 0: feed `first_token` (the argmax of the last prompt position) into
    // decode forward. The hidden produced is the hidden *at the position of
    // first_token* (since the model produces logits *for the next token* at
    // each forward, but scratch.hidden is the post-final-layer hidden at the
    // current step). prev_token == last_prompt_token (it's what came before
    // first_token in the sequence).
    let mut current_input_token = first_token;
    let mut prev_input_token = *prompt_tokens.last().unwrap();
    for step in 0..decode_count {
        let (hidden, logits) = engine
            .debug_decode_next_hidden_and_logits(current_input_token)
            .expect("debug_decode_next_hidden_and_logits failed");
        assert_eq!(
            hidden.len(),
            hidden_dim,
            "step {step}: hidden len {} != hidden_dim {}",
            hidden.len(),
            hidden_dim
        );
        let argmax = if logits.is_empty() {
            engine
                .last_backend_argmax_token()
                .expect("empty logits without backend argmax")
        } else {
            rnb_llm::sampler::greedy::greedy_sample(&logits)
        };
        hiddens.push(hidden);
        argmax_tokens.push(argmax);
        prev_tokens.push(prev_input_token);

        prev_input_token = current_input_token;
        current_input_token = argmax;
    }

    // Write the binary file.
    let mut f = std::fs::File::create(dump_path).expect("create trace dump file");
    f.write_all(b"RNBHTRC1").unwrap();
    f.write_all(&(hidden_dim as u32).to_le_bytes()).unwrap();
    f.write_all(&(hiddens.len() as u32).to_le_bytes()).unwrap();
    for ((h, &argmax), &prev) in hiddens
        .iter()
        .zip(argmax_tokens.iter())
        .zip(prev_tokens.iter())
    {
        for &v in h {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.write_all(&argmax.to_le_bytes()).unwrap();
        f.write_all(&prev.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
    eprintln!(
        "[TRACE_DUMP] wrote {} positions × hidden_dim={} → {}",
        hiddens.len(),
        hidden_dim,
        dump_path
    );

    // Brief sanity log: first 10 argmax tokens + decoded pieces.
    let preview = argmax_tokens
        .iter()
        .take(10)
        .map(|&t| format!("{}={:?}", t, engine.tokenizer.decode_token(t)))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("[TRACE_DUMP] first argmax pieces: {preview}");
}

fn load_prompt_text(target_len: usize, fallback: &str) -> String {
    if let Ok(prompt) = std::env::var("RNB_PROMPT") {
        return if target_len > 0 {
            prompt.chars().take(target_len * 8).collect()
        } else {
            prompt
        };
    }
    if let Ok(path) = std::env::var("RNB_PROMPT_FILE") {
        let text = std::fs::read_to_string(&path).expect("Failed to read prompt file");
        return if target_len > 0 {
            text.chars().take(target_len * 8).collect()
        } else {
            text
        };
    }
    fallback.to_string()
}

#[derive(Default)]
struct GemmaChatRenderState {
    suppress_until_channel_close: bool,
    hide_channel_markup: bool,
    /// When true, render every non-markup piece including the thinking
    /// channel contents. Set via `RNB_GEMMA_SHOW_THINKING=1`.
    show_thinking: bool,
}

fn render_chat_piece(
    piece: &str,
    gemma_chat: bool,
    state: &mut GemmaChatRenderState,
) -> Option<String> {
    if !gemma_chat {
        return Some(piece.to_string());
    }

    // Show-thinking mode (RNB_GEMMA_SHOW_THINKING=1): render every piece
    // unchanged so the caller sees both the thinking channel and the final
    // answer interleaved as the model emits them. Channel markers are still
    // visible, which is the point.
    if state.show_thinking {
        return Some(piece.to_string());
    }

    if piece == "<|channel>" {
        state.hide_channel_markup = true;
        return None;
    }
    if piece == "<channel|>" {
        if state.suppress_until_channel_close {
            state.suppress_until_channel_close = false;
        }
        state.hide_channel_markup = false;
        return None;
    }
    if state.hide_channel_markup || state.suppress_until_channel_close {
        return None;
    }
    if piece == "<|turn>" || piece == "<turn|>" || piece == "<|think|>" {
        return None;
    }
    Some(piece.to_string())
}

fn extract_gemma_final_answer(raw: &str) -> Option<String> {
    let mut candidate = raw;
    if let Some((_, tail)) = raw.rsplit_once("<channel|>") {
        candidate = tail;
    }
    if let Some((_, tail)) = candidate.rsplit_once("<turn|>") {
        candidate = tail;
    }

    let cleaned = candidate
        .replace("<|channel>", "")
        .replace("<channel|>", "")
        .replace("<|turn>", "")
        .replace("<turn|>", "")
        .replace("<|think|>", "")
        .replace('{', "")
        .replace('}', "")
        .trim()
        .to_string();

    (!cleaned.is_empty()).then_some(cleaned)
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn cuda_exact_greedy_auto_allowed_for_backend(
    spec_mode: Option<&str>,
    chat_mode: bool,
    exact_greedy_chat_arch: bool,
    repetition_penalty: f32,
    has_target_pieces: bool,
    chat_min_content_tokens_set: bool,
    output_argmax_set: bool,
    suppress_special_set: bool,
    suppress_pieces_set: bool,
    dump_topk_set: bool,
    dump_topk_decode_set: bool,
    backend_output_argmax_supported: bool,
) -> bool {
    if matches!(spec_mode, Some("1" | "2")) {
        return false;
    }
    backend_output_argmax_supported
        && (!chat_mode || exact_greedy_chat_arch)
        && repetition_penalty == 1.0
        && !has_target_pieces
        && !chat_min_content_tokens_set
        && !output_argmax_set
        && !suppress_special_set
        && !suppress_pieces_set
        && !dump_topk_set
        && !dump_topk_decode_set
}

#[cfg(any(feature = "metal", test))]
#[allow(clippy::too_many_arguments)]
fn metal_exact_greedy_auto_allowed_for_backend(
    spec_mode: Option<&str>,
    chat_mode: bool,
    exact_greedy_chat_arch: bool,
    repetition_penalty: f32,
    has_target_pieces: bool,
    chat_min_content_tokens_set: bool,
    output_argmax_set: bool,
    suppress_special_set: bool,
    suppress_pieces_set: bool,
    dump_topk_set: bool,
    dump_topk_decode_set: bool,
    backend_output_argmax_supported: bool,
) -> bool {
    cuda_exact_greedy_auto_allowed_for_backend(
        spec_mode,
        chat_mode,
        exact_greedy_chat_arch,
        repetition_penalty,
        has_target_pieces,
        chat_min_content_tokens_set,
        output_argmax_set,
        suppress_special_set,
        suppress_pieces_set,
        dump_topk_set,
        dump_topk_decode_set,
        backend_output_argmax_supported,
    )
}

fn standard_generate_mode_requested(spec_mode: Option<&str>, generate_env_set: bool) -> bool {
    generate_env_set && spec_mode.is_none()
}

fn mtp_generate_mode_requested(
    spec_mode: Option<&str>,
    mtp_env_requested: bool,
    generate_env_set: bool,
) -> bool {
    if matches!(spec_mode, Some("2")) {
        return false;
    }
    mtp_env_requested || standard_generate_mode_requested(spec_mode, generate_env_set)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MtpEnvRequest {
    Off,
    Force,
    Auto,
}

fn parse_mtp_env_request(raw: Option<&str>) -> MtpEnvRequest {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return MtpEnvRequest::Off;
    };
    let lower = raw.to_ascii_lowercase();
    match lower.as_str() {
        "0" | "false" | "off" | "no" => MtpEnvRequest::Off,
        "auto" => MtpEnvRequest::Auto,
        _ => MtpEnvRequest::Force,
    }
}

fn mtp_env_requests_generation(
    request: MtpEnvRequest,
    policy: rnb_llm::engine::MtpAutoPolicy,
) -> bool {
    match request {
        MtpEnvRequest::Off => false,
        MtpEnvRequest::Force => true,
        MtpEnvRequest::Auto => policy.enabled,
    }
}

fn mtp_env_off_policy() -> rnb_llm::engine::MtpAutoPolicy {
    rnb_llm::engine::MtpAutoPolicy {
        enabled: false,
        spec_k: 4,
        device_verify: false,
        min_free_vram_mib: 0,
        resource: None,
        reason: "mtp-env-off",
    }
}

fn mtp_should_enable_device_verify(
    request: MtpEnvRequest,
    policy: rnb_llm::engine::MtpAutoPolicy,
    device_verify_env_set: bool,
) -> bool {
    !device_verify_env_set && policy.device_verify && mtp_env_requests_generation(request, policy)
}

fn resolve_mtp_spec_k(
    raw: Option<&str>,
    policy: rnb_llm::engine::MtpAutoPolicy,
) -> Result<usize, String> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(policy.spec_k.max(1));
    };
    if raw.eq_ignore_ascii_case("auto") {
        return Ok(policy.spec_k.max(1));
    }
    let value = raw
        .parse::<usize>()
        .map_err(|_| format!("RNB_SPEC_K must be a positive integer or auto, got {raw:?}"))?;
    if value == 0 {
        return Err("RNB_SPEC_K must be >= 1".to_string());
    }
    Ok(value)
}

fn parse_mtp_abab_repeat(raw: Option<&str>) -> Result<Option<usize>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let repeat = raw
        .trim()
        .parse::<usize>()
        .map_err(|_| format!("RNB_MTP_ABAB_REPEAT must be an even integer >= 2, got {raw:?}"))?;
    if repeat < 2 || repeat % 2 != 0 {
        return Err(format!(
            "RNB_MTP_ABAB_REPEAT must be an even integer >= 2, got {repeat}"
        ));
    }
    Ok(Some(repeat))
}

fn parse_mtp_abab_spec_k_b(raw: Option<&str>) -> Result<Option<usize>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let spec_k = raw
        .trim()
        .parse::<usize>()
        .map_err(|_| format!("RNB_MTP_ABAB_SPEC_K_B must be a positive integer, got {raw:?}"))?;
    if spec_k == 0 {
        return Err("RNB_MTP_ABAB_SPEC_K_B must be >= 1".to_string());
    }
    Ok(Some(spec_k))
}

fn validate_mtp_abab_timing_env(
    profile_enabled: bool,
    spec_profile_enabled: bool,
    trace_enabled: bool,
) -> Result<(), &'static str> {
    if profile_enabled || spec_profile_enabled || trace_enabled {
        return Err(
            "RNB_MTP_ABAB_REPEAT requires timer-I/O-free runs; unset RNB_PROFILE, RNB_SPEC_PROFILE, and RNB_MTP_TRACE",
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MtpAbabVariant {
    Sequential,
    BatchPrefill,
}

impl MtpAbabVariant {
    fn for_run(run_index: usize) -> Self {
        if run_index % 2 == 0 {
            Self::Sequential
        } else {
            Self::BatchPrefill
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Sequential => "A/Sequential",
            Self::BatchPrefill => "B/BatchPrefill",
        }
    }

    fn batch_verify(self) -> &'static str {
        match self {
            Self::Sequential => "0",
            Self::BatchPrefill => "1",
        }
    }
}

fn median_ms(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[middle - 1] + sorted[middle]) / 2.0)
    } else {
        Some(sorted[middle])
    }
}

fn mtp_abab_medians(a_ms: &[f64], b_ms: &[f64]) -> (Option<f64>, Option<f64>) {
    (
        median_ms(a_ms.get(1..).unwrap_or_default()),
        median_ms(b_ms),
    )
}

fn mtp_abab_result_equality(
    canonical: Option<&(String, usize)>,
    text: &str,
    tokens_generated: usize,
) -> Option<bool> {
    canonical.map(|(canonical_text, canonical_tokens)| {
        canonical_text == text && *canonical_tokens == tokens_generated
    })
}

fn mtp_abab_continue(_piece: &str) -> bool {
    true
}

fn mtp_abab_draft_only_requested(raw: Option<&str>) -> bool {
    raw.is_some_and(|value| {
        let value = value.to_ascii_lowercase();
        !matches!(value.as_str(), "0" | "false" | "off" | "no")
    })
}

fn validate_mtp_abab_path(
    mtp_requested: bool,
    spec_mode: Option<&str>,
    draft_only: bool,
    has_weights: bool,
    has_mtp: bool,
    architecture: rnb_loader::Architecture,
) -> Result<(), &'static str> {
    if spec_mode == Some("2") {
        return Err(
            "RNB_MTP_ABAB_REPEAT requires the in-model MTP path; RNB_SPEC=2 selects two-model speculative decoding",
        );
    }
    if !mtp_requested {
        return Err(
            "RNB_MTP_ABAB_REPEAT requires the in-model MTP generate path (set RNB_MTP=1 or use an enabled RNB_MTP=auto policy)",
        );
    }
    if draft_only {
        return Err(
            "RNB_MTP_ABAB_REPEAT cannot use RNB_MTP_DRAFT_ONLY because verify toggles are bypassed",
        );
    }
    if !has_weights {
        return Err("RNB_MTP_ABAB_REPEAT requires loaded model weights");
    }
    if !has_mtp {
        return Err("RNB_MTP_ABAB_REPEAT requires a ready in-model MTP runtime and weights");
    }
    if architecture == rnb_loader::Architecture::Gemma4 {
        return Err(
            "RNB_MTP_ABAB_REPEAT does not support the Gemma4 external drafter runtime because it does not consume the batch verify toggle",
        );
    }
    Ok(())
}

struct EnvVarRestore {
    name: &'static str,
    value: Option<std::ffi::OsString>,
}

impl EnvVarRestore {
    fn capture(name: &'static str) -> Self {
        Self {
            name,
            value: std::env::var_os(name),
        }
    }
    fn set_temporarily(name: &'static str, value: &str) -> Self {
        let restore = Self::capture(name);
        // SAFETY: dev-cli 벤치는 모델 로드와 측정을 단일 스레드에서 순차 실행하며,
        // 반환 guard가 main의 모든 정상/오류 종료에서 원래 값을 복원한다.
        unsafe {
            std::env::set_var(name, value);
        }
        restore
    }
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        // SAFETY: dev-cli 벤치는 단일 스레드 측정 하니스이며 env 변경 구간이 guard 수명으로 제한된다.
        unsafe {
            match &self.value {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }
}

fn run_mtp_abab(
    engine: &mut rnb_llm::Engine,
    prompt: &str,
    params: &rnb_llm::generate::GenerateParams,
    repeat: usize,
) {
    let spec_k_b = parse_mtp_abab_spec_k_b(std::env::var("RNB_MTP_ABAB_SPEC_K_B").ok().as_deref())
        .unwrap_or_else(|err| panic!("{err}"));
    let _batch_verify_restore = EnvVarRestore::capture("RNB_MTP_BATCH_VERIFY");
    let _sequential_multi_restore =
        spec_k_b.map(|_| EnvVarRestore::set_temporarily("RNB_SPEC_MTP_SEQUENTIAL_MULTI", "1"));
    assert_eq!(
        std::env::var("RNB_MTP_DEVICE_VERIFY").as_deref(),
        Ok("0"),
        "RNB_MTP_ABAB_REPEAT must force RNB_MTP_DEVICE_VERIFY=0 before model load"
    );
    let mut a_ms = Vec::with_capacity(repeat.div_ceil(2));
    let mut b_ms = Vec::with_capacity(repeat / 2);
    let mut canonical: Option<(String, usize)> = None;

    let comparison = if spec_k_b.is_some() {
        "spec-k"
    } else {
        "batch-verify"
    };
    eprintln!(
        "[MTP_ABAB] start runs={repeat} order=AB comparison={comparison} device_verify=0 max_tokens={} a_spec_k={} b_spec_k={} temperature={}",
        params.max_tokens,
        params.spec_k,
        spec_k_b.unwrap_or(params.spec_k),
        params.temperature
    );

    for run_index in 0..repeat {
        let variant = MtpAbabVariant::for_run(run_index);
        let candidate = variant == MtpAbabVariant::BatchPrefill;
        let variant_label = match (spec_k_b.is_some(), candidate) {
            (true, false) => "A/SpecKBaseline",
            (true, true) => "B/SpecKCandidate",
            (false, _) => variant.label(),
        };
        let batch_verify = if spec_k_b.is_some() {
            "0"
        } else {
            variant.batch_verify()
        };
        let mut run_params = params.clone();
        if candidate {
            if let Some(spec_k) = spec_k_b {
                run_params.spec_k = spec_k;
            }
        }
        // SAFETY: dev-cli 벤치는 단일 스레드로 각 generate 호출 전에 env 를 설정한다.
        unsafe {
            std::env::set_var("RNB_MTP_BATCH_VERIFY", batch_verify);
        }

        let started = std::time::Instant::now();
        let generated = engine.generate_stream(prompt, &run_params, mtp_abab_continue);
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let result = match generated {
            Ok(result) => result,
            Err(err) => {
                eprintln!(
                    "[MTP_ABAB] run={} variant={variant_label} status=error wall_ms={wall_ms:.3} error={err}",
                    run_index + 1,
                );
                panic!(
                    "MTP ABAB run {} ({variant_label}) failed: {err}",
                    run_index + 1,
                );
            }
        };

        let generated_chars = result.text.chars().count();
        let generated_sha256 = generated_text_sha256_hex(&result.text);
        let equality =
            mtp_abab_result_equality(canonical.as_ref(), &result.text, result.tokens_generated);
        let equality_label = match equality {
            None => "reference",
            Some(true) => "true",
            Some(false) => "false",
        };
        eprintln!(
            "[MTP_ABAB] run={} variant={variant_label} batch_verify={batch_verify} spec_k={} device_verify=0 cold={} wall_ms={wall_ms:.3} generated_tokens={} generated_chars={generated_chars} text_sha256={generated_sha256} equality={equality_label}",
            run_index + 1,
            run_params.spec_k,
            run_index == 0,
            result.tokens_generated,
        );
        if equality == Some(false) {
            panic!(
                "MTP ABAB output mismatch at run {} ({variant_label})",
                run_index + 1,
            );
        }
        if canonical.is_none() {
            canonical = Some((result.text, result.tokens_generated));
        }
        match variant {
            MtpAbabVariant::Sequential => a_ms.push(wall_ms),
            MtpAbabVariant::BatchPrefill => b_ms.push(wall_ms),
        }
    }

    let (a_median_ms, b_median_ms) = mtp_abab_medians(&a_ms, &b_ms);
    let a_median_ms = a_median_ms
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "NA".to_string());
    let b_median_ms = b_median_ms
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "NA".to_string());
    eprintln!(
        "[MTP_ABAB] summary runs={repeat} excluded=A1 a_samples={} b_samples={} a_median_ms={a_median_ms} b_median_ms={b_median_ms}",
        a_ms.len().saturating_sub(1),
        b_ms.len()
    );
}

fn decode_piece_output_enabled(quiet_decode: bool) -> bool {
    !quiet_decode
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerifyMicrobenchConfig {
    window: usize,
    rounds: usize,
}

fn parse_verify_microbench_config(
    window_raw: Option<&str>,
    rounds_raw: Option<&str>,
) -> VerifyMicrobenchConfig {
    let window = window_raw
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(2)
        .max(1);
    let rounds = rounds_raw
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    VerifyMicrobenchConfig { window, rounds }
}

fn greedy_token_from_logits_or_backend(engine: &rnb_llm::Engine, logits: &[f32]) -> u32 {
    if logits.is_empty() {
        engine
            .last_backend_argmax_token()
            .expect("empty logits without backend argmax token")
    } else {
        rnb_llm::sampler::greedy::greedy_sample(logits)
    }
}

fn reset_engine_to_prompt(engine: &mut rnb_llm::Engine, tokens: &[u32]) -> Vec<f32> {
    engine.clear_sequence_state().unwrap();
    run_prefill(engine, tokens)
}

fn collect_greedy_verify_tokens(
    engine: &mut rnb_llm::Engine,
    prompt_tokens: &[u32],
    window: usize,
) -> Vec<u32> {
    let logits = reset_engine_to_prompt(engine, prompt_tokens);
    let mut token = greedy_token_from_logits_or_backend(engine, &logits);
    let mut verify_tokens = Vec::with_capacity(window);
    verify_tokens.push(token);
    for _ in 1..window {
        let logits = engine.forward(&[token]).unwrap();
        token = greedy_token_from_logits_or_backend(engine, &logits);
        verify_tokens.push(token);
    }
    verify_tokens
}

fn run_verify_microbench(
    engine: &mut rnb_llm::Engine,
    prompt_tokens: &[u32],
    config: VerifyMicrobenchConfig,
) {
    let verify_tokens = collect_greedy_verify_tokens(engine, prompt_tokens, config.window);
    eprintln!(
        "[VERIFY_MICRO] window={} rounds={} prompt_tokens={}",
        config.window,
        config.rounds,
        prompt_tokens.len(),
    );
    dump_tokens(engine, &verify_tokens, "verify_micro");

    let mut sequential_ms = 0.0;
    for _ in 0..config.rounds {
        reset_engine_to_prompt(engine, prompt_tokens);
        let start = std::time::Instant::now();
        for &token in &verify_tokens {
            let _ = engine.forward(&[token]).unwrap();
        }
        sequential_ms += start.elapsed().as_secs_f64() * 1000.0;
    }

    let mut batch_ms = 0.0;
    for _ in 0..config.rounds {
        reset_engine_to_prompt(engine, prompt_tokens);
        let start = std::time::Instant::now();
        let _ = engine.forward_prefill_all_logits(&verify_tokens).unwrap();
        batch_ms += start.elapsed().as_secs_f64() * 1000.0;
    }

    let rounds = config.rounds as f64;
    let tokens = config.window as f64 * rounds;
    eprintln!(
        "[VERIFY_MICRO] sequential_decode total_ms={:.3} per_round_ms={:.3} per_token_ms={:.3}",
        sequential_ms,
        sequential_ms / rounds,
        sequential_ms / tokens,
    );
    eprintln!(
        "[VERIFY_MICRO] batch_all_logits total_ms={:.3} per_round_ms={:.3} per_token_ms={:.3} ratio_vs_seq={:.3}",
        batch_ms,
        batch_ms / rounds,
        batch_ms / tokens,
        if sequential_ms > 0.0 {
            batch_ms / sequential_ms
        } else {
            0.0
        },
    );
    print_profile_reports();
}

fn run_affinity_sweep(raw: &str) {
    let specs = parse_affinity_sweep_specs(raw);
    if specs.is_empty() {
        eprintln!("[WARN] RNB_AFFINITY_SWEEP is empty; running single benchmark");
        return;
    }

    let exe = std::env::current_exe().expect("failed to resolve current executable");
    eprintln!("=== Affinity sweep: {} ===", specs.join(", "));

    for spec in specs {
        eprintln!("\n=== Affinity run: {spec} ===");
        let start = std::time::Instant::now();
        let status = std::process::Command::new(&exe)
            .env("RNB_CPU_AFFINITY", &spec)
            .env_remove("RNB_AFFINITY_SWEEP")
            .status()
            .expect("failed to spawn affinity sweep child");
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "=== Affinity run done: {spec} status={} wall={:.0}ms ===",
            status, wall_ms
        );
        assert!(status.success(), "affinity sweep child failed for {spec}");
    }
}

fn build_gemma_chat_prompt_text(prompt: &str) -> String {
    let system_prompt = std::env::var("RNB_SYSTEM_PROMPT").unwrap_or_default();
    let include_think = std::env::var("RNB_GEMMA_INCLUDE_THINK")
        .map(|v| v != "0")
        .unwrap_or(false);
    let minimal_chat = std::env::var("RNB_GEMMA_MINIMAL_CHAT").is_ok();
    let user_role = std::env::var("RNB_GEMMA_USER_ROLE").unwrap_or_else(|_| "user".to_string());
    let model_role = std::env::var("RNB_GEMMA_MODEL_ROLE").unwrap_or_else(|_| "model".to_string());

    let mut text = String::new();

    if !minimal_chat && (include_think || !system_prompt.trim().is_empty()) {
        text.push_str("<|turn>system\n");
        if include_think {
            text.push_str("<|think|>\n");
        }
        if !system_prompt.trim().is_empty() {
            text.push_str(&system_prompt);
        }
        text.push_str("<turn|>\n");
    }

    text.push_str(&format!("<|turn>{user_role}\n"));
    text.push_str(prompt);
    text.push_str("<turn|>\n");
    text.push_str(&format!("<|turn>{model_role}\n"));
    text
}

fn build_gemma_chat_tokens(engine: &rnb_llm::Engine, bos: u32, prompt: &str) -> Vec<u32> {
    // BOS policy is owned by the tokenizer (GGUF
    // `tokenizer.ggml.add_bos_token`). Gemma4 sets it to true and falls into
    // garbage decoding without it; Qwen3.5 sets it to false and is fine
    // either way. The legacy `RNB_NO_BOS=1` override has been retired —
    // there is no scenario where forcing the wrong policy is the right
    // answer.
    let mut tokens = if engine.tokenizer.should_add_bos() {
        vec![bos]
    } else {
        Vec::new()
    };
    let text = build_gemma_chat_prompt_text(prompt);
    tokens.extend(engine.tokenizer.encode(&text));
    tokens
}

fn gemma_chat_stop_tokens(engine: &rnb_llm::Engine) -> Vec<u32> {
    let mut out = vec![engine.tokenizer.vocab.special.eos];
    if let Some(id) = engine.tokenizer.vocab.token_id("<turn|>") {
        out.push(id);
    }
    if let Some(id) = engine.tokenizer.vocab.token_id("</s>") {
        out.push(id);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn nemotron_env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(default)
}

fn nemotron_thinking_enabled(prompt: &str) -> bool {
    nemotron_thinking_enabled_with_default(
        prompt,
        nemotron_env_flag("RNB_NEMOTRON_ENABLE_THINKING", true),
    )
}

fn nemotron_thinking_enabled_with_default(prompt: &str, default: bool) -> bool {
    let mut enabled = default;
    let sanitized = prompt.replace("</think>", "");
    if sanitized.contains("/think") {
        enabled = true;
    }
    if prompt.contains("/no_think") {
        enabled = false;
    }
    enabled
}

fn sanitize_nemotron_thinking_directives(text: &str) -> String {
    text.replace("</think>", "<_end_think>")
        .replace("/think", "")
        .replace("/no_think", "")
        .replace("<_end_think>", "</think>")
}

#[allow(dead_code)]
fn build_nemotron_chat_prompt_text(prompt: &str) -> String {
    let system_prompt = std::env::var("RNB_SYSTEM_PROMPT").unwrap_or_default();
    let thinking = nemotron_thinking_enabled(prompt);
    build_nemotron_chat_prompt_text_with_options(prompt, &system_prompt, thinking)
}

#[cfg_attr(not(test), allow(dead_code))]
fn build_nemotron_chat_prompt_text_with_options(
    prompt: &str,
    system_prompt: &str,
    thinking: bool,
) -> String {
    let user_prompt = sanitize_nemotron_thinking_directives(prompt)
        .trim()
        .to_string();
    let mut text = String::new();
    text.push_str("<|im_start|>system\n");
    if !system_prompt.trim().is_empty() {
        text.push_str(sanitize_nemotron_thinking_directives(&system_prompt).trim());
    }
    text.push_str("<|im_end|>\n");
    text.push_str("<|im_start|>user\n");
    text.push_str(&user_prompt);
    text.push_str("<|im_end|>\n");
    if thinking {
        text.push_str("<|im_start|>assistant\n<think>\n");
    } else {
        text.push_str("<|im_start|>assistant\n<think></think>");
    }
    text
}

fn build_nemotron_chat_tokens(engine: &rnb_llm::Engine, prompt: &str) -> Vec<u32> {
    let system_prompt = std::env::var("RNB_SYSTEM_PROMPT").unwrap_or_default();
    let thinking = nemotron_thinking_enabled(prompt);
    let user_prompt = sanitize_nemotron_thinking_directives(prompt)
        .trim()
        .to_string();
    let system_prompt = sanitize_nemotron_thinking_directives(&system_prompt)
        .trim()
        .to_string();
    let im_start = engine
        .tokenizer
        .vocab
        .token_id("<|im_start|>")
        .expect("missing <|im_start|>");
    let im_end = engine
        .tokenizer
        .vocab
        .token_id("<|im_end|>")
        .expect("missing <|im_end|>");
    let nl = engine
        .tokenizer
        .vocab
        .token_id("\n")
        .or_else(|| engine.tokenizer.vocab.token_id("Ċ"))
        .expect("missing newline token");

    let mut tokens = Vec::new();
    tokens.push(im_start);
    tokens.extend(engine.tokenizer.encode("system"));
    tokens.push(nl);
    if !system_prompt.is_empty() {
        tokens.extend(engine.tokenizer.encode(&system_prompt));
    }
    tokens.push(im_end);
    tokens.push(nl);

    tokens.push(im_start);
    tokens.extend(engine.tokenizer.encode("user"));
    tokens.push(nl);
    tokens.extend(engine.tokenizer.encode(&user_prompt));
    tokens.push(im_end);
    tokens.push(nl);

    tokens.push(im_start);
    tokens.extend(engine.tokenizer.encode("assistant"));
    tokens.push(nl);
    push_exact_or_encoded(engine, &mut tokens, "<think>");
    if thinking {
        tokens.push(nl);
    } else {
        push_exact_or_encoded(engine, &mut tokens, "</think>");
    }
    tokens
}

fn nemotron_chat_stop_tokens(engine: &rnb_llm::Engine) -> Vec<u32> {
    let mut out = vec![engine.tokenizer.vocab.special.eos];
    if let Some(id) = engine.tokenizer.vocab.token_id("<|im_end|>") {
        out.push(id);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn push_exact_or_encoded(engine: &rnb_llm::Engine, tokens: &mut Vec<u32>, text: &str) {
    if let Some(id) = engine.tokenizer.vocab.token_id(text) {
        tokens.push(id);
    } else {
        tokens.extend(engine.tokenizer.encode(text));
    }
}

/// Qwen3.5 GDN per-layer profiling
/// cu59 axis A — RNB_CU58_DIAG_CHAIN 활성 시 process 종료 직전 chain function
/// sub-phase aggregate 출력. drop 순서로 main 함수가 어떻게 빠져나가든 (정상 return,
/// early return, panic) 자동 호출.
struct ChainDiagAggregateOnExit;
impl Drop for ChainDiagAggregateOnExit {
    fn drop(&mut self) {
        rnb_llm::dump_chain_diag_aggregate();
    }
}

fn main() {
    if llm_bench_help_requested(std::env::args().skip(1)) {
        println!("{}", llm_bench_usage());
        return;
    }

    let _cu59_diag_guard = ChainDiagAggregateOnExit;
    #[cfg(feature = "mediatek")]
    if mediatek_quant_gated_gelu_probe_requested() {
        if let Err(err) = run_mtk_quant_gated_gelu_probe_from_env() {
            eprintln!("[mediatek-quant-ffn-probe] error={err}");
            std::process::exit(1);
        }
        return;
    }

    if let Ok(raw) = std::env::var("RNB_AFFINITY_SWEEP") {
        run_affinity_sweep(&raw);
        return;
    }

    let wall_start = std::time::Instant::now();
    let affinity_label = effective_affinity_label(
        std::env::var("RNB_CPU_AFFINITY").ok().as_deref(),
        std::env::var("RNB_BIG_CORES").is_ok(),
    );
    eprintln!("[INFO] llm-bench affinity mode: {}", affinity_label);

    if std::env::var("RNB_GPU_FULLPATH")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        eprintln!("[INFO] mode=GPU_FULLPATH (mv20)");
    }

    let mtp_abab_repeat =
        parse_mtp_abab_repeat(std::env::var("RNB_MTP_ABAB_REPEAT").ok().as_deref())
            .unwrap_or_else(|err| panic!("{err}"));
    if mtp_abab_repeat.is_some() {
        validate_mtp_abab_timing_env(
            std::env::var_os("RNB_PROFILE").is_some(),
            std::env::var_os("RNB_SPEC_PROFILE").is_some(),
            std::env::var_os("RNB_MTP_TRACE").is_some(),
        )
        .unwrap_or_else(|err| panic!("{err}"));
    }
    // Device verify affects persistent CUDA cache policy during Engine construction,
    // so ABAB must disable it before loading rather than only before generation.
    let _mtp_abab_device_verify_restore =
        mtp_abab_repeat.map(|_| EnvVarRestore::set_temporarily("RNB_MTP_DEVICE_VERIFY", "0"));

    let path = std::path::PathBuf::from(
        std::env::var("RNB_MODEL")
            .unwrap_or_else(|_| "models/Qwen3.5-0.8B-Q4_K_M.gguf".to_string()),
    );
    let load_start = std::time::Instant::now();
    let diagnostic_sidecar =
        std::env::var_os("RNB_DIAGNOSTIC_SIDECAR").map(std::path::PathBuf::from);
    if let Some(sidecar) = diagnostic_sidecar.as_deref() {
        eprintln!(
            "[rnb-llm-bench] diagnostic sidecar explicitly requested: {}",
            sidecar.display()
        );
    }
    let host_ram_budget_bytes = host_ram_budget_bytes_from_env().unwrap_or_else(|error| {
        eprintln!("[rnb-llm-bench] {error}");
        std::process::exit(2);
    });
    let load_config =
        rnb_llm::EngineLoadConfig::default().with_diagnostic_sidecar(diagnostic_sidecar);
    let load_config = match host_ram_budget_bytes {
        Some(bytes) => load_config.with_host_ram_budget_bytes(bytes),
        None => load_config,
    };
    let mut engine = rnb_llm::Engine::from_gguf_with_config(&path, load_config).unwrap();
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    if std::env::var("RNB_GEMV_PROFILE").is_ok() {
        rnb_llm::engine::reset_gemv_profile();
    }
    if std::env::var("RNB_MOE_PROFILE").is_ok() {
        rnb_llm::engine::reset_moe_profile();
    }
    rnb_llm::engine::reset_packed_dispatch();

    // Prefill — RNB_PROMPT_FILE로 텍스트 파일 지정, RNB_PREFILL_TOKENS로 토큰 수 제한
    let bos = engine.tokenizer.vocab.special.bos;
    let target_len: usize = std::env::var("RNB_PREFILL_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let prompt_text = load_prompt_text(
        target_len,
        "The quick brown fox jumps over the lazy dog. This is a longer prompt to test prefill performance with more tokens. \
         We want to measure how well the weight-stationary GEMM optimization works for realistic conversation lengths. \
         In a typical chat scenario, the prompt would include system instructions, conversation history, and the user's latest message. \
         This gives us a reasonable approximation of real-world prefill workloads.",
    );
    let prompt_tokens = engine.tokenizer.encode(&prompt_text);
    let target_pieces = parse_target_pieces_env();
    // RNB_CHAT=1 이면 Qwen/Qwen3.5 chat template 적용.
    let chat_mode = std::env::var("RNB_CHAT").is_ok();

    // special token IDs를 미리 가져옴 (borrow 해제용)
    let im_start_id = engine.tokenizer.vocab.token_id("<|im_start|>");
    let im_end_id = engine.tokenizer.vocab.token_id("<|im_end|>");
    let nl_id = engine
        .tokenizer
        .vocab
        .token_id("\n")
        .or_else(|| engine.tokenizer.vocab.token_id("Ċ"));
    let eos_id = im_end_id.unwrap_or(engine.tokenizer.vocab.special.eos);
    let gemma_turn_open_id = engine.tokenizer.vocab.token_id("<|turn>");
    let gemma_turn_close_id = engine.tokenizer.vocab.token_id("<turn|>");
    let gemma_think_id = engine.tokenizer.vocab.token_id("<|think|>");

    let mut tokens = Vec::new();
    let gemma_chat = chat_mode
        && gemma_turn_open_id.is_some()
        && gemma_turn_close_id.is_some()
        && gemma_think_id.is_some();
    let gemma_arch = matches!(
        engine.architecture(),
        rnb_loader::Architecture::Gemma | rnb_loader::Architecture::Gemma4
    );
    let exact_greedy_chat_arch = gemma_chat || gemma_arch;
    let nemotron_chat =
        chat_mode && engine.architecture() == rnb_loader::Architecture::NemotronHMoE;
    let default_repetition_penalty = if chat_mode && !exact_greedy_chat_arch {
        1.1
    } else {
        1.0
    };
    let repetition_penalty = std::env::var("RNB_REPETITION_PENALTY")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default_repetition_penalty);
    if chat_mode {
        if nemotron_chat {
            tokens = build_nemotron_chat_tokens(&engine, &prompt_text);
        } else if gemma_chat {
            tokens = build_gemma_chat_tokens(&engine, bos, &prompt_text);
        } else {
            let im_start = im_start_id.expect("missing <|im_start|>");
            let im_end = im_end_id.expect("missing <|im_end|>");
            let nl = nl_id.expect("missing newline token");
            tokens.push(im_start);
            tokens.extend(engine.tokenizer.encode("user"));
            tokens.push(nl);
            tokens.extend(&prompt_tokens);
            tokens.push(im_end);
            tokens.push(nl);
            tokens.push(im_start);
            tokens.extend(engine.tokenizer.encode("assistant"));
            tokens.push(nl);
            if std::env::var("RNB_QWEN_ENABLE_THINKING").is_ok() {
                tokens.extend(engine.tokenizer.encode("<think>"));
                tokens.push(nl);
            } else {
                tokens.extend(engine.tokenizer.encode("<think>"));
                tokens.push(nl);
                tokens.push(nl);
                tokens.extend(engine.tokenizer.encode("</think>"));
                tokens.push(nl);
                tokens.push(nl);
            }
        }
    } else {
        // BOS policy is owned by the GGUF tokenizer metadata
        // (`tokenizer.ggml.add_bos_token`). Models that need BOS (Gemma4)
        // get it; models that don't (Qwen3.5) skip it. No env override —
        // forcing the wrong policy was the cause of Gemma4 garbage decode
        // in earlier benches.
        if engine.tokenizer.should_add_bos() {
            tokens.push(bos);
        }
        tokens.extend(&prompt_tokens);
    }
    if target_len > 0 && tokens.len() > target_len {
        tokens.truncate(target_len);
    }
    if std::env::var("RNB_DUMP_TOKENS").is_ok() {
        dump_tokens(&engine, &tokens, "prompt");
    }

    // Decode tokens with per-forward timing (RNB_DECODE_TOKENS, default 20)
    let decode_count: usize = std::env::var("RNB_DECODE_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    // --- 단일 프로세스 prefill ABAB 반복 측정 (D축 GDN prefill chain wall) ----
    // RNB_PREFILL_ABAB_REPEAT=N 이면 모델/엔진 1회 로드 후 같은 prefill 을 ABAB 로
    // 2N 회 반복(A=chain ON, B=chain OFF), 매 prefill 전 state reset. decode 안 함.
    // 다른 모든 path 보다 먼저 분기.
    if let Ok(raw) = std::env::var("RNB_PREFILL_ABAB_REPEAT") {
        let repeat: usize = raw.parse().unwrap_or(0);
        if repeat == 0 {
            eprintln!(
                "[ABAB] RNB_PREFILL_ABAB_REPEAT={raw} 파싱 실패 또는 0 — N>=1 이어야 함. 종료."
            );
            return;
        }
        run_prefill_abab(&mut engine, &tokens, repeat);
        return;
    }

    // --- Calibration trace dump (Stage C-2, drafter calibration) ----
    // RNB_DUMP_BACKBONE_HIDDEN=<path> 면 target-only greedy decode 를 돌리고
    // 매 position 의 (normed_backbone_hidden, argmax_token, prev_token) 를
    // binary 로 dump. 다른 모든 path 보다 먼저 분기해서 단순 경로로만 진행.
    if let Ok(dump_path) = std::env::var("RNB_DUMP_BACKBONE_HIDDEN") {
        dump_backbone_hidden_trace(&mut engine, &tokens, decode_count, &dump_path);
        return;
    }
    let spec_mode = std::env::var("RNB_SPEC").ok();
    if std::env::var("RNB_VERIFY_MICRO").is_ok() {
        let config = parse_verify_microbench_config(
            std::env::var("RNB_VERIFY_MICRO_WINDOW").ok().as_deref(),
            std::env::var("RNB_VERIFY_MICRO_ROUNDS").ok().as_deref(),
        );
        run_verify_microbench(&mut engine, &tokens, config);
        return;
    }

    use rnb_llm::generate::GenerateParams;
    // mc78: RNB_MTP_DRAFT_N controls external drafter draft size. RNB_SPEC_K
    // is the legacy in-model nextn knob; for external drafter generate path
    // RNB_MTP_DRAFT_N takes precedence (and falls back to spec_k if unset).
    let mtp_draft_n_override: Option<usize> = std::env::var("RNB_MTP_DRAFT_N")
        .ok()
        .and_then(|s| s.parse().ok());
    let mtp_env_request = parse_mtp_env_request(std::env::var("RNB_MTP").ok().as_deref());
    let mtp_policy = if matches!(mtp_env_request, MtpEnvRequest::Off) {
        mtp_env_off_policy()
    } else {
        engine.mtp_auto_policy()
    };
    let mtp_env_requested = mtp_env_requests_generation(mtp_env_request, mtp_policy);
    if mtp_abab_repeat.is_some() {
        let draft_only =
            mtp_abab_draft_only_requested(std::env::var("RNB_MTP_DRAFT_ONLY").ok().as_deref());
        validate_mtp_abab_path(
            mtp_env_requested,
            spec_mode.as_deref(),
            draft_only,
            engine.has_weights(),
            engine.has_mtp(),
            engine.architecture(),
        )
        .unwrap_or_else(|err| panic!("{err}"));
    }
    let mtp_set_device_verify = mtp_should_enable_device_verify(
        mtp_env_request,
        mtp_policy,
        std::env::var("RNB_MTP_DEVICE_VERIFY").is_ok(),
    );
    if matches!(mtp_env_request, MtpEnvRequest::Auto) {
        let resource = mtp_policy
            .resource
            .map(|resource| {
                format!(
                    " total_vram={}MiB free_vram={}MiB",
                    resource.total_vram_mib, resource.free_vram_mib
                )
            })
            .unwrap_or_default();
        eprintln!(
            "[MTP_AUTO] enabled={} k={} device_verify={} min_free_vram={}MiB{} reason={}",
            mtp_policy.enabled,
            mtp_policy.spec_k,
            mtp_policy.device_verify,
            mtp_policy.min_free_vram_mib,
            resource,
            mtp_policy.reason
        );
        if mtp_policy.enabled {
            std::env::set_var("RNB_MTP", "1");
        } else {
            std::env::set_var("RNB_MTP", "0");
        }
    }
    if mtp_set_device_verify && mtp_abab_repeat.is_none() {
        std::env::set_var("RNB_MTP_DEVICE_VERIFY", "1");
    }
    let generate_env_set = std::env::var("RNB_GENERATE").is_ok();

    // --- Speculative decoding mode ---
    if spec_mode.as_deref() == Some("1") && !mtp_env_requested {
        let spec_k: usize = std::env::var("RNB_SPEC_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let spec_depth: f32 = std::env::var("RNB_SPEC_DEPTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);

        let params = GenerateParams {
            max_tokens: decode_count,
            temperature: 0.0,
            spec_enabled: true,
            spec_k,
            spec_depth,
            ..GenerateParams::default()
        };

        eprintln!(
            "[SPEC] k={}, depth={:.0}%, decode_tokens={}",
            spec_k,
            spec_depth * 100.0,
            decode_count
        );

        let emit_decode_pieces =
            decode_piece_output_enabled(std::env::var("RNB_QUIET_DECODE").is_ok());
        let spec_start = std::time::Instant::now();
        let result = engine
            .generate_stream(&prompt_text, &params, |piece| {
                if emit_decode_pieces {
                    eprint!("{}", piece);
                }
                true
            })
            .unwrap();
        dump_generated_text_hash_if_requested(&result);
        let spec_ms = spec_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "\n[SPEC] prompt={} tokens, generated={} tokens, wall={:.0}ms ({:.1} tok/s including prefill)",
            result.prompt_tokens,
            result.tokens_generated,
            spec_ms,
            if spec_ms > 0.0 {
                result.tokens_generated as f64 / (spec_ms / 1000.0)
            } else {
                0.0
            },
        );
        if std::env::var("RNB_BENCH_WALL").is_ok() {
            eprintln!(
                "RNB_BENCH_WALL load_ms={:.3} spec_ms={:.3} total_ms={:.3} prompt_tokens={} decode_tokens={}",
                load_ms,
                spec_ms,
                load_ms + spec_ms,
                result.prompt_tokens,
                result.tokens_generated,
            );
        }
        print_profile_reports();
        return;
    }

    // --- Two-model speculative decoding mode ---
    if spec_mode.as_deref() == Some("2") {
        let spec_k: usize = std::env::var("RNB_SPEC_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);

        let draft_model_path = std::env::var("RNB_DRAFT_MODEL")
            .expect("RNB_SPEC=2 requires RNB_DRAFT_MODEL=path/to/draft.gguf");

        eprintln!("[SPEC2] Loading draft model: {}", draft_model_path);
        let mut draft_engine =
            rnb_llm::Engine::from_gguf(&std::path::PathBuf::from(&draft_model_path)).unwrap();

        let params = GenerateParams {
            max_tokens: decode_count,
            temperature: 0.0,
            spec_enabled: true,
            spec_k,
            spec_depth: 1.0,
            ..GenerateParams::default()
        };

        eprintln!("[SPEC2] k={}, decode_tokens={}", spec_k, decode_count);

        let emit_decode_pieces =
            decode_piece_output_enabled(std::env::var("RNB_QUIET_DECODE").is_ok());
        let spec_start = std::time::Instant::now();
        let result = rnb_llm::speculative::generate_stream_two_model(
            &mut draft_engine,
            &mut engine,
            &prompt_text,
            &params,
            |piece| {
                if emit_decode_pieces {
                    eprint!("{}", piece);
                }
                true
            },
        )
        .unwrap();
        dump_generated_text_hash_if_requested(&result);
        let spec_ms = spec_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "\n[SPEC2] prompt={} tokens, generated={} tokens, wall={:.0}ms ({:.1} tok/s including prefill)",
            result.prompt_tokens,
            result.tokens_generated,
            spec_ms,
            if spec_ms > 0.0 {
                result.tokens_generated as f64 / (spec_ms / 1000.0)
            } else {
                0.0
            },
        );
        if std::env::var("RNB_BENCH_WALL").is_ok() {
            eprintln!(
                "RNB_BENCH_WALL load_ms={:.3} spec_ms={:.3} total_ms={:.3} prompt_tokens={} decode_tokens={}",
                load_ms,
                spec_ms,
                load_ms + spec_ms,
                result.prompt_tokens,
                result.tokens_generated,
            );
        }
        print_profile_reports();
        return;
    }

    if mtp_generate_mode_requested(spec_mode.as_deref(), mtp_env_requested, generate_env_set) {
        let spec_k = if let Some(n) = mtp_draft_n_override {
            n.max(1)
        } else {
            resolve_mtp_spec_k(std::env::var("RNB_SPEC_K").ok().as_deref(), mtp_policy)
                .unwrap_or_else(|err| panic!("{err}"))
        };
        let spec_depth: f32 = std::env::var("RNB_SPEC_DEPTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);
        let params = GenerateParams {
            max_tokens: decode_count,
            temperature: 0.0,
            repetition_penalty,
            spec_k,
            spec_depth,
            ..GenerateParams::default()
        };
        let quiet_generate = std::env::var("RNB_QUIET_DECODE").is_ok();
        if mtp_env_requested {
            eprintln!(
                "[MTP] decode_tokens={}, k={}, depth={:.0}%",
                decode_count,
                spec_k,
                spec_depth * 100.0
            );
        } else {
            eprintln!("[GENERATE] decode_tokens={}", decode_count);
        }

        // mc78: chat mode 일 때 generate_stream/mtp 가 받는 prompt 도 raw decode
        // bench 의 build_gemma_chat_tokens 와 동일하게 chat template wrap 후
        // 넘긴다. generate_stream 안에서 tokenizer.encode 가 같은 토큰 시퀀스
        // 를 만들도록 보장.
        let wrapped_prompt: String;
        let effective_prompt: &str = if gemma_chat {
            wrapped_prompt = build_gemma_chat_prompt_text(&prompt_text);
            &wrapped_prompt
        } else {
            &prompt_text
        };
        if let Some(repeat) = mtp_abab_repeat {
            run_mtp_abab(&mut engine, effective_prompt, &params, repeat);
            print_profile_reports();
            return;
        }
        let generate_start = std::time::Instant::now();
        let result = engine
            .generate_stream(effective_prompt, &params, |piece| {
                if !quiet_generate {
                    eprint!("{}", piece);
                }
                true
            })
            .unwrap();
        dump_generated_text_hash_if_requested(&result);
        let generate_ms = generate_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "\n[GENERATE] prompt={} tokens, generated={} tokens, wall={:.0}ms ({:.1} tok/s including prefill)",
            result.prompt_tokens,
            result.tokens_generated,
            generate_ms,
            if generate_ms > 0.0 {
                result.tokens_generated as f64 / (generate_ms / 1000.0)
            } else {
                0.0
            },
        );
        if std::env::var("RNB_BENCH_WALL").is_ok() {
            eprintln!(
                "RNB_BENCH_WALL load_ms={:.3} generate_ms={:.3} total_ms={:.3} prompt_tokens={} decode_tokens={}",
                load_ms,
                generate_ms,
                load_ms + generate_ms,
                result.prompt_tokens,
                result.tokens_generated,
            );
        }
        print_profile_reports();
        return;
    }

    let backend_output_argmax_supported = engine.backend_output_argmax_supported_for_runtime();

    #[cfg(feature = "cuda")]
    if cuda_exact_greedy_auto_allowed_for_backend(
        spec_mode.as_deref(),
        chat_mode,
        exact_greedy_chat_arch,
        repetition_penalty,
        !target_pieces.is_empty(),
        std::env::var("RNB_CHAT_MIN_CONTENT_TOKENS").is_ok(),
        std::env::var("RNB_CUDA_OUTPUT_ARGMAX").is_ok(),
        std::env::var("RNB_SUPPRESS_SPECIAL_TOKENS").is_ok(),
        std::env::var("RNB_SUPPRESS_PIECES").is_ok(),
        std::env::var("RNB_DUMP_TOPK").is_ok(),
        std::env::var("RNB_DUMP_TOPK_DECODE").is_ok(),
        backend_output_argmax_supported,
    ) {
        std::env::set_var("RNB_CUDA_OUTPUT_ARGMAX", "1");
        if std::env::var("RNB_CUDA_PREFILL_ARGMAX_ONLY").is_err() {
            std::env::set_var("RNB_CUDA_PREFILL_ARGMAX_ONLY", "1");
        }
        std::env::set_var("RNB_CUDA_PREFILL_OUTPUT_LOGITS", "1");
        eprintln!("[INFO] CUDA exact greedy prefill/decode argmax enabled");
    }
    #[cfg(feature = "metal")]
    if metal_exact_greedy_auto_allowed_for_backend(
        spec_mode.as_deref(),
        chat_mode,
        exact_greedy_chat_arch,
        repetition_penalty,
        !target_pieces.is_empty(),
        std::env::var("RNB_CHAT_MIN_CONTENT_TOKENS").is_ok(),
        std::env::var("RNB_METAL_OUTPUT_ARGMAX").is_ok(),
        std::env::var("RNB_SUPPRESS_SPECIAL_TOKENS").is_ok(),
        std::env::var("RNB_SUPPRESS_PIECES").is_ok(),
        std::env::var("RNB_DUMP_TOPK").is_ok(),
        std::env::var("RNB_DUMP_TOPK_DECODE").is_ok(),
        backend_output_argmax_supported,
    ) {
        std::env::set_var("RNB_METAL_OUTPUT_ARGMAX", "1");
        eprintln!("[INFO] Metal exact greedy decode argmax enabled");
    }
    #[cfg(feature = "cuda")]
    if std::env::var("RNB_CUDA_OUTPUT_ARGMAX").is_ok() && engine.prewarm_output_weight_for_runtime()
    {
        eprintln!("[INFO] CUDA output weight prewarmed");
    }
    #[cfg(feature = "cuda")]
    if let Some(warmup_len) = std::env::var("RNB_CUDA_PREFILL_WARMUP_TOKENS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&len| len > 0)
        .map(|len| len.min(tokens.len()))
        .filter(|&len| len > 0)
    {
        let warmup_start = std::time::Instant::now();
        let warmup_tokens = &tokens[..warmup_len];
        let _ = run_prefill(&mut engine, warmup_tokens);
        engine.clear_sequence_state().unwrap();
        eprintln!(
            "[INFO] CUDA prefill warmup: tokens={} elapsed_ms={:.3}",
            warmup_len,
            warmup_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    eprintln!("Prefill {} tokens", tokens.len());
    let start = std::time::Instant::now();
    let logits = run_prefill(&mut engine, &tokens);
    if let Ok(raw) = std::env::var("RNB_DUMP_TOPK") {
        if let Ok(k) = raw.parse::<usize>() {
            dump_top_logits(&engine, &logits, k, "prefill_last");
        }
    }
    if !target_pieces.is_empty() {
        let target_refs = target_pieces.iter().map(String::as_str).collect::<Vec<_>>();
        dump_target_ranks(&engine, &logits, &target_refs, "prefill_last");
    }
    let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Prefill: {:.0}ms ({:.1} tok/s)",
        prefill_ms,
        tokens.len() as f64 / (prefill_ms / 1000.0)
    );
    #[cfg(feature = "vulkan")]
    if let Some(counters) = engine.prefill_runtime_counters() {
        eprintln!(
            "{}",
            format_vulkan_runtime_counters(
                counters.submits,
                counters.upload_bytes,
                counters.download_bytes,
                counters.materializations,
                counters.attention_fan_in_copies,
            )
        );
    } else {
        eprintln!("[vulkan:counters] unavailable");
    }

    // eos_id는 위에서 미리 계산됨

    // Chat 모드: SamplerChain 사용 (repetition_penalty + temperature + min_p)
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rnb_llm::sampler::SamplerChain;

    let params = if chat_mode {
        GenerateParams {
            repetition_penalty,
            temperature: 0.0,
            ..GenerateParams::default()
        }
    } else {
        GenerateParams {
            repetition_penalty,
            temperature: 0.0,
            ..GenerateParams::default()
        }
    };
    let mut sampler = SamplerChain::from_params(&params);
    let mut rng = SmallRng::seed_from_u64(42);
    // repetition penalty는 생성된 토큰에만 적용 (프롬프트 제외)
    let gemma_stop_tokens = if nemotron_chat {
        nemotron_chat_stop_tokens(&engine)
    } else if gemma_chat {
        gemma_chat_stop_tokens(&engine)
    } else {
        vec![eos_id]
    };
    let mut generated_tokens: Vec<u32> = Vec::new();
    let decode_topk_steps = std::env::var("RNB_DUMP_TOPK_DECODE_STEPS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(1);
    let decode_target_steps = std::env::var("RNB_DUMP_TARGET_DECODE_STEPS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(decode_topk_steps);
    let show_thinking = std::env::var("RNB_GEMMA_SHOW_THINKING").is_ok();
    let mut gemma_render = GemmaChatRenderState {
        // In show-thinking mode we never suppress anything, so start open.
        suppress_until_channel_close: gemma_chat && !show_thinking,
        hide_channel_markup: false,
        show_thinking,
    };
    let mut gemma_raw_output = String::new();
    let default_chat_min_content_tokens = if chat_mode
        && !exact_greedy_chat_arch
        && std::env::var("RNB_CHAT_ALLOW_IMMEDIATE_EOS").is_err()
    {
        16usize
    } else {
        0usize
    };
    let chat_min_content_tokens = std::env::var("RNB_CHAT_MIN_CONTENT_TOKENS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(default_chat_min_content_tokens);

    let mut first_logits = logits;
    let mut token = if first_logits.is_empty() {
        engine
            .last_backend_argmax_token()
            .expect("fullpath prefill returned empty logits without backend_argmax_token")
    } else {
        suppress_special_logits(&engine, &mut first_logits, chat_min_content_tokens > 0);
        suppress_selected_pieces(&engine, &mut first_logits);
        if let Ok(raw) = std::env::var("RNB_DUMP_TOPK") {
            if let Ok(k) = raw.parse::<usize>() {
                dump_top_logits(&engine, &first_logits, k, "prefill_sample");
            }
        }
        sampler.sample(&mut first_logits, &generated_tokens, &mut rng)
    };
    generated_tokens.push(token);

    #[cfg(feature = "cuda")]
    if (!chat_mode || exact_greedy_chat_arch)
        && repetition_penalty == 1.0
        && chat_min_content_tokens == 0
        && target_pieces.is_empty()
        && backend_output_argmax_supported
        && std::env::var("RNB_CUDA_OUTPUT_ARGMAX").is_err()
        && std::env::var("RNB_SUPPRESS_SPECIAL_TOKENS").is_err()
        && std::env::var("RNB_SUPPRESS_PIECES").is_err()
        && std::env::var("RNB_DUMP_TOPK_DECODE").is_err()
    {
        std::env::set_var("RNB_CUDA_OUTPUT_ARGMAX", "1");
        eprintln!("[INFO] CUDA exact greedy decode: output argmax enabled");
    }
    let backend_output_argmax_enabled = std::env::var("RNB_CUDA_OUTPUT_ARGMAX").as_deref()
        == Ok("1")
        || std::env::var("RNB_METAL_OUTPUT_ARGMAX").as_deref() == Ok("1");
    let exact_backend_argmax_decode = (!chat_mode || exact_greedy_chat_arch)
        && repetition_penalty == 1.0
        && chat_min_content_tokens == 0
        && target_pieces.is_empty()
        && backend_output_argmax_supported
        && backend_output_argmax_enabled
        && std::env::var("RNB_SUPPRESS_SPECIAL_TOKENS").is_err()
        && std::env::var("RNB_SUPPRESS_PIECES").is_err()
        && std::env::var("RNB_DUMP_TOPK_DECODE").is_err();

    rnb_llm::reset_metal_decode_parity_counters();
    let decode_start = std::time::Instant::now();
    let mut decode_tokens = 0u32;
    let quiet_decode = std::env::var("RNB_QUIET_DECODE").is_ok();
    if decode_count > 0 {
        if !quiet_decode {
            let tok_str = engine.tokenizer.decode_token(token);
            if chat_mode {
                if gemma_chat {
                    gemma_raw_output.push_str(&tok_str);
                    if let Some(rendered) =
                        render_chat_piece(&tok_str, gemma_chat, &mut gemma_render)
                    {
                        use std::io::Write;
                        eprint!("{}", rendered);
                        let _ = std::io::stderr().flush();
                    }
                } else if let Some(rendered) =
                    render_chat_piece(&tok_str, gemma_chat, &mut gemma_render)
                {
                    eprint!("{}", rendered);
                }
            } else {
                eprintln!("Decode 0: 0.0ms → {:?}", tok_str);
            }
        }
        decode_tokens += 1;
    }
    for step in 1..decode_count {
        if chat_mode && gemma_stop_tokens.contains(&token) {
            eprintln!("[EOS at step {}]", step);
            break;
        }
        let start = std::time::Instant::now();
        let backend_argmax_token = if exact_backend_argmax_decode {
            engine.forward_decode_backend_argmax_only(token).unwrap()
        } else {
            None
        };
        let mut logits = if backend_argmax_token.is_some() {
            Vec::new()
        } else {
            let mut logits = engine.forward(&[token]).unwrap();
            let force_suppress_special = generated_tokens.len() < chat_min_content_tokens;
            suppress_special_logits(&engine, &mut logits, force_suppress_special);
            suppress_selected_pieces(&engine, &mut logits);
            if step <= decode_topk_steps {
                if let Ok(raw) = std::env::var("RNB_DUMP_TOPK_DECODE") {
                    if let Ok(k) = raw.parse::<usize>() {
                        let label = format!("decode_step{}", step - 1);
                        dump_top_logits(&engine, &logits, k, &label);
                    }
                }
            }
            if step <= decode_target_steps {
                if !target_pieces.is_empty() {
                    let target_refs = target_pieces.iter().map(String::as_str).collect::<Vec<_>>();
                    let label = format!("decode_step{}", step - 1);
                    dump_target_ranks(&engine, &logits, &target_refs, &label);
                }
            }
            logits
        };
        let elapsed_us = start.elapsed().as_micros();
        token = if let Some(token) = backend_argmax_token {
            token
        } else if logits.is_empty() {
            engine
                .last_backend_argmax_token()
                .expect("fullpath decode returned empty logits without backend_argmax_token")
        } else {
            sampler.sample(&mut logits, &generated_tokens, &mut rng)
        };
        generated_tokens.push(token);
        decode_tokens += 1;
        if !quiet_decode {
            let tok_str = engine.tokenizer.decode_token(token);
            if chat_mode {
                if gemma_chat {
                    gemma_raw_output.push_str(&tok_str);
                    if let Some(rendered) =
                        render_chat_piece(&tok_str, gemma_chat, &mut gemma_render)
                    {
                        use std::io::Write;
                        eprint!("{}", rendered);
                        let _ = std::io::stderr().flush();
                    }
                } else if let Some(rendered) =
                    render_chat_piece(&tok_str, gemma_chat, &mut gemma_render)
                {
                    eprint!("{}", rendered);
                }
            } else {
                eprintln!(
                    "Decode {}: {:.1}ms → {:?}",
                    step,
                    elapsed_us as f64 / 1000.0,
                    tok_str
                );
            }
        }
    }
    let decode_elapsed = decode_start.elapsed();
    rnb_llm::report_metal_decode_parity_counters("decode");
    if std::env::var("RNB_BENCH_WALL").is_ok() {
        let total_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
        let decode_ms = decode_elapsed.as_secs_f64() * 1000.0;
        eprintln!(
            "RNB_BENCH_WALL load_ms={:.3} prefill_ms={:.3} decode_ms={:.3} total_ms={:.3} prompt_tokens={} decode_tokens={}",
            load_ms,
            prefill_ms,
            decode_ms,
            total_ms,
            tokens.len(),
            decode_tokens
        );
    }
    if std::env::var("RNB_DUMP_GENERATED_TOKENS").is_ok() {
        dump_tokens(&engine, &generated_tokens, "generated");
    }
    if chat_mode {
        eprintln!(); // 줄바꿈
        if gemma_chat {
            if let Some(final_answer) = extract_gemma_final_answer(&gemma_raw_output) {
                eprintln!("[gemma-final] {}", final_answer);
            }
        }
        eprintln!("\n--- Chat stats ---");
        eprintln!(
            "Prefill: {} tokens, {:.0}ms ({:.1} tok/s)",
            tokens.len(),
            prefill_ms,
            tokens.len() as f64 / (prefill_ms / 1000.0)
        );
        eprintln!(
            "Decode: {} tokens, {:.0}ms ({:.1} tok/s)",
            decode_tokens,
            decode_elapsed.as_millis(),
            decode_tokens as f64 / decode_elapsed.as_secs_f64()
        );
        eprintln!(
            "Total: {:.0}ms",
            prefill_ms + decode_elapsed.as_millis() as f64
        );
        if let Some((n_calls, total_us, total_bytes)) = engine.cold_reader_stats() {
            let n_tokens = (tokens.len() as u64) + (decode_tokens as u64);
            eprintln!(
                "Cold pread stats: {} calls, {:.1} ms total, {:.2} MB total, {:.1} calls/tok, {:.2} ms/tok",
                n_calls,
                total_us as f64 / 1000.0,
                total_bytes as f64 / (1024.0 * 1024.0),
                n_calls as f64 / n_tokens as f64,
                (total_us as f64 / 1000.0) / n_tokens as f64,
            );
        }
        if let Some(report) = rnb_llm::engine::gemv_profile_report() {
            eprintln!("\n{}", report);
        }
        if let Some(report) = rnb_llm::engine::moe_profile_report() {
            eprintln!("\n{}", report);
        }
        if let Some(report) = rnb_llm::engine::packed_dispatch_report() {
            eprintln!("\n{}", report);
        }
    }

    if chat_mode || std::env::var("RNB_SKIP_DECODE_BENCH").is_ok() {
        if !chat_mode {
            if let Some(report) = rnb_llm::engine::gemv_profile_report() {
                eprintln!("\n{}", report);
            }
            if let Some(report) = rnb_llm::engine::moe_profile_report() {
                eprintln!("\n{}", report);
            }
        }
        if let Some(report) = rnb_llm::engine::moe_jit_report() {
            eprintln!("\n{}", report);
        }
        if let Some(report) = rnb_llm::engine::packed_dispatch_report() {
            eprintln!("\n{}", report);
        }
        return;
    }

    // Now profile individual components
    eprintln!("\n=== Component profiling (1 decode step) ===");

    // We'll time by running forward with RUST_LOG or by measuring each layer type
    // For now, let's count how many of each layer type
    let meta = &engine.metadata;
    let interval = meta.full_attention_interval;
    let n_layers = meta.num_layers;

    let n_gdn = (0..n_layers)
        .filter(|&i| interval > 0 && i % interval != (interval - 1))
        .count();
    let n_attn = n_layers - n_gdn;

    eprintln!(
        "Layers: {} total ({} GDN + {} attention)",
        n_layers, n_gdn, n_attn
    );
    // 27B(qwen35) 실측 GEMV shape [N=out rows, K=in dim], nb=K/256. (GGUF Qwen3.6-27B-Q4_K_M,
    // hidden=5120, ffn=17408, ssm_inner=6144). 전부 K∈{5120,6144,17408} → nb∈{20,24,68}:
    // 현 P1(gemv_q4k_simd) sub_block 조건(pow2 && 2≤nb≤32) 불만족 → 전부 stride fallback
    // (nb=20 → `for b=lane;b<20;b+=32` → 20 lane active, 12 idle = 37% idle).
    eprintln!("GDN layer GEMV [N, K] nb=K/256 (all stride fallback):");
    eprintln!("  attn_qkv  [10240,  5120] nb=20 Q6_K");
    eprintln!("  attn_gate [ 6144,  5120] nb=20 Q4_K");
    eprintln!("  ssm_alpha [   48,  5120] nb=20 F32 (tiny N → low occupancy)");
    eprintln!("  ssm_beta  [   48,  5120] nb=20 F32 (tiny N → low occupancy)");
    eprintln!("  ssm_out   [ 5120,  6144] nb=24 Q5_K");
    eprintln!("  ffn_gate  [17408,  5120] nb=20 Q4_K");
    eprintln!("  ffn_up    [17408,  5120] nb=20 Q4_K");
    eprintln!("  ffn_down  [ 5120, 17408] nb=68 Q6_K");
    eprintln!("attn layer GEMV: attn_q [12288,5120] nb20 Q4_K, attn_k/v [1024,5120] nb20,");
    eprintln!("  attn_output [5120,6144] nb24 Q4_K, ffn = GDN 와 동일.");

    // Profile raw gemv speed
    use std::time::Instant;
    let _dummy = rnb_core::tensor::Tensor::from_slice(&vec![0.1f32; 1024], &[1, 1024]);

    // Test: time 18 GDN qkv gemvs (the biggest operation)
    // We need access to weights, but they're private. Let's just time the full forward.

    if std::env::var("RNB_SKIP_DECODE_BENCH").is_ok() {
        return;
    }
    if std::env::var("RNB_MOE_PROFILE").is_ok() {
        rnb_llm::engine::reset_moe_profile();
    }

    if std::env::var("RNB_DIRTY_PACKED_AB").is_ok() {
        let rounds: usize = std::env::var("RNB_DIRTY_PACKED_AB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let old = std::env::var("RNB_PACKED_DECODE").ok();
        eprintln!("\n=== dirty packed decode AB ({} rounds) ===", rounds);
        for round in 0..rounds {
            let packed_first = round % 2 == 1;
            let mut run_once = |packed: bool, token: u32| {
                if packed {
                    std::env::set_var("RNB_PACKED_DECODE", "1");
                } else {
                    std::env::remove_var("RNB_PACKED_DECODE");
                }
                let start = Instant::now();
                let logits = engine.forward(&[token]).unwrap();
                let us = start.elapsed().as_micros();
                let next = if logits.is_empty() {
                    engine.last_backend_argmax_token().expect(
                        "fullpath decode returned empty logits without backend_argmax_token",
                    )
                } else {
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as u32)
                        .unwrap()
                };
                (us, next)
            };

            let (base_us, base_next, packed_us, packed_next) = if packed_first {
                let (packed_us, packed_next) = run_once(true, token);
                let (base_us, base_next) = run_once(false, token);
                (base_us, base_next, packed_us, packed_next)
            } else {
                let (base_us, base_next) = run_once(false, token);
                let (packed_us, packed_next) = run_once(true, token);
                (base_us, base_next, packed_us, packed_next)
            };

            eprintln!(
                "  round {} {}: base={:.1}ms packed={:.1}ms ratio={:.2}x next=({},{})",
                round,
                if packed_first { "BA" } else { "AB" },
                base_us as f64 / 1000.0,
                packed_us as f64 / 1000.0,
                packed_us as f64 / base_us.max(1) as f64,
                base_next,
                packed_next
            );
            token = base_next;
        }
        match old {
            Some(v) => std::env::set_var("RNB_PACKED_DECODE", v),
            None => std::env::remove_var("RNB_PACKED_DECODE"),
        }
        if std::env::var("RNB_DIRTY_PACKED_AB_ONLY").is_ok() {
            return;
        }
    }

    // Time N decode steps (RNB_BENCH_DECODE_STEPS, default 10; 0 = skip)
    let bench_steps: usize = std::env::var("RNB_BENCH_DECODE_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    if bench_steps > 0 {
        eprintln!("\n=== {} decode steps ===", bench_steps);
        let mut total_us = 0u128;
        let trace_freq = cpu_freq_trace_enabled();
        for step in 0..bench_steps {
            if trace_freq {
                eprintln!("  freq {} before: {}", step, cpu_freq_snapshot());
            }
            let start = Instant::now();
            let logits = engine.forward(&[token]).unwrap();
            let us = start.elapsed().as_micros();
            total_us += us;
            token = if logits.is_empty() {
                engine
                    .last_backend_argmax_token()
                    .expect("fullpath decode returned empty logits without backend_argmax_token")
            } else {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            eprintln!("  step {}: {:.1}ms", step, us as f64 / 1000.0);
            if trace_freq {
                eprintln!("  freq {} after: {}", step, cpu_freq_snapshot());
            }
        }
        eprintln!(
            "Average: {:.1}ms/token ({:.1} tok/s)",
            total_us as f64 / bench_steps as f64 / 1000.0,
            bench_steps as f64 * 1_000_000.0 / total_us as f64
        );
    }

    let warm_reruns: usize = std::env::var("RNB_WARM_RERUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if warm_reruns > 0 {
        eprintln!("\n=== {} warm reruns ===", warm_reruns);
        let reference_tokens = generated_tokens.clone();
        for run_idx in 0..warm_reruns {
            let request_start = std::time::Instant::now();
            let reset_start = std::time::Instant::now();
            engine.clear_sequence_state().unwrap();
            let reset_ms = reset_start.elapsed().as_secs_f64() * 1000.0;

            let prefill_start = std::time::Instant::now();
            let mut warm_logits = run_prefill(&mut engine, &tokens);
            let warm_prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

            let mut warm_sampler = SamplerChain::from_params(&params);
            let mut warm_rng = SmallRng::seed_from_u64(42);
            let mut warm_generated_tokens: Vec<u32> = Vec::new();

            let mut warm_token = if decode_count > 0 {
                if warm_logits.is_empty() {
                    Some(engine.last_backend_argmax_token().expect(
                        "fullpath prefill returned empty logits without backend_argmax_token",
                    ))
                } else {
                    suppress_special_logits(&engine, &mut warm_logits, chat_min_content_tokens > 0);
                    suppress_selected_pieces(&engine, &mut warm_logits);
                    Some(warm_sampler.sample(
                        &mut warm_logits,
                        &warm_generated_tokens,
                        &mut warm_rng,
                    ))
                }
            } else {
                None
            };

            let warm_decode_start = std::time::Instant::now();
            if let Some(mut current_token) = warm_token.take() {
                warm_generated_tokens.push(current_token);
                for _step in 1..decode_count {
                    if chat_mode && gemma_stop_tokens.contains(&current_token) {
                        break;
                    }
                    let backend_argmax_token = if exact_backend_argmax_decode {
                        engine
                            .forward_decode_backend_argmax_only(current_token)
                            .unwrap()
                    } else {
                        None
                    };
                    let mut logits = if backend_argmax_token.is_some() {
                        Vec::new()
                    } else {
                        let mut logits = engine.forward(&[current_token]).unwrap();
                        let force_suppress_special =
                            warm_generated_tokens.len() < chat_min_content_tokens;
                        suppress_special_logits(&engine, &mut logits, force_suppress_special);
                        suppress_selected_pieces(&engine, &mut logits);
                        logits
                    };
                    current_token = if let Some(token) = backend_argmax_token {
                        token
                    } else if logits.is_empty() {
                        engine.last_backend_argmax_token().expect(
                            "fullpath decode returned empty logits without backend_argmax_token",
                        )
                    } else {
                        warm_sampler.sample(&mut logits, &warm_generated_tokens, &mut warm_rng)
                    };
                    warm_generated_tokens.push(current_token);
                }
            }
            let warm_decode_ms = warm_decode_start.elapsed().as_secs_f64() * 1000.0;
            let request_total_ms = request_start.elapsed().as_secs_f64() * 1000.0;
            let token_match = warm_generated_tokens == reference_tokens;
            eprintln!(
                "RNB_WARM_RERUN run={} reset_ms={:.3} prefill_ms={:.3} decode_ms={:.3} request_total_ms={:.3} prompt_tokens={} decode_tokens={} token_match={}",
                run_idx + 1,
                reset_ms,
                warm_prefill_ms,
                warm_decode_ms,
                request_total_ms,
                tokens.len(),
                warm_generated_tokens.len(),
                token_match
            );
            if !token_match {
                let diff_idx = reference_tokens
                    .iter()
                    .zip(warm_generated_tokens.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| reference_tokens.len().min(warm_generated_tokens.len()));
                let expected = reference_tokens
                    .get(diff_idx)
                    .map(|id| (*id, engine.tokenizer.decode_token(*id)))
                    .unwrap_or((u32::MAX, "<missing>".to_string()));
                let actual = warm_generated_tokens
                    .get(diff_idx)
                    .map(|id| (*id, engine.tokenizer.decode_token(*id)))
                    .unwrap_or((u32::MAX, "<missing>".to_string()));
                eprintln!(
                    "RNB_WARM_RERUN_DIFF run={} index={} expected_id={} expected_piece={:?} actual_id={} actual_piece={:?}",
                    run_idx + 1,
                    diff_idx,
                    expected.0,
                    expected.1.replace('\n', "\\n"),
                    actual.0,
                    actual.1.replace('\n', "\\n")
                );
            }
        }
    }

    if let Some((n_calls, total_us, total_bytes)) = engine.cold_reader_stats() {
        eprintln!(
            "Cold pread stats: {} calls, {:.1} ms total, {:.2} MB total",
            n_calls,
            total_us as f64 / 1000.0,
            total_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    if let Some(report) = rnb_llm::engine::gemv_profile_report() {
        eprintln!("\n{}", report);
    }
    if let Some(report) = rnb_llm::engine::moe_profile_report() {
        eprintln!("\n{}", report);
    }
    if let Some(report) = rnb_llm::engine::moe_jit_report() {
        eprintln!("\n{}", report);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_bench_help_requested_accepts_short_and_long_flags() {
        assert!(llm_bench_help_requested(["--help"]));
        assert!(llm_bench_help_requested(["-h"]));
        assert!(llm_bench_help_requested(["--ignored", "--help"]));
        assert!(!llm_bench_help_requested(["--ignored"]));
    }

    #[test]
    fn test_parse_affinity_sweep_specs_splits_semicolons() {
        assert_eq!(
            parse_affinity_sweep_specs("all;big;little;0,2,7"),
            vec!["all", "big", "little", "0,2,7"]
        );
    }

    #[test]
    fn test_parse_affinity_sweep_specs_ignores_empty_segments() {
        assert_eq!(
            parse_affinity_sweep_specs("all;; big ; "),
            vec!["all", "big"]
        );
    }

    #[test]
    fn test_effective_affinity_label_prefers_explicit_value() {
        assert_eq!(effective_affinity_label(Some("little"), false), "little");
    }

    #[test]
    fn test_effective_affinity_label_uses_legacy_big_flag() {
        assert_eq!(effective_affinity_label(None, true), "big(legacy)");
    }

    #[test]
    fn host_ram_budget_unset_keeps_automatic_policy() {
        assert_eq!(parse_host_ram_budget_bytes(None), Ok(None));
    }

    #[test]
    fn host_ram_budget_zero_explicitly_clears_the_budget() {
        let bytes = parse_host_ram_budget_bytes(Some("0")).unwrap().unwrap();
        assert_eq!(bytes, 0);
        assert!(rnb_llm::EngineLoadConfig::default()
            .with_host_ram_budget_bytes(bytes)
            .host_memory_budget
            .is_none());
    }

    #[test]
    fn host_ram_budget_accepts_positive_decimal_u64_bytes() {
        assert_eq!(
            parse_host_ram_budget_bytes(Some("34359738368")),
            Ok(Some(34_359_738_368))
        );
        assert_eq!(
            parse_host_ram_budget_bytes(Some("18446744073709551615")),
            Ok(Some(u64::MAX))
        );
    }

    #[test]
    fn host_ram_budget_rejects_malformed_values() {
        for raw in ["", "16GiB", "-1", "+1", " 1024", "18446744073709551616"] {
            let error = parse_host_ram_budget_bytes(Some(raw)).unwrap_err();
            assert!(error.contains(HOST_RAM_BUDGET_ENV));
            assert!(error.contains("decimal u64 bytes"));
        }
    }

    #[test]
    fn test_format_vulkan_runtime_counters_includes_all_fields() {
        assert_eq!(
            format_vulkan_runtime_counters(3, 1024, 2048, 1, 0),
            "[vulkan:counters] submits=3 upload_bytes=1024 download_bytes=2048 materializations=1 attention_fan_in_copies=0"
        );
    }

    #[test]
    fn test_format_profile_reports_joins_present_reports() {
        assert_eq!(
            format_profile_reports(Some("gemv".to_string()), Some("moe".to_string())),
            Some("gemv\nmoe".to_string())
        );
        assert_eq!(
            format_profile_reports(None, Some("moe".to_string())),
            Some("moe".to_string())
        );
        assert_eq!(format_profile_reports(None, None), None);
    }

    #[test]
    fn prefill_abab_mode_rejects_multiple_explicit_modes() {
        let mode = prefill_abab_mode_from_lookup(|var| match var {
            "RNB_PREFILL_ABAB_ATTN_CHAIN"
            | "RNB_PREFILL_ABAB_ATN_FULL_LAYER"
            | "RNB_PREFILL_ABAB_ATN_O_TAIL" => Some("1".to_string()),
            _ => None,
        });

        let err = mode.expect_err("multiple ABAB modes must be rejected");
        assert!(err.contains("RNB_PREFILL_ABAB_ATTN_CHAIN"));
        assert!(err.contains("RNB_PREFILL_ABAB_ATN_FULL_LAYER"));
        assert!(err.contains("RNB_PREFILL_ABAB_ATN_O_TAIL"));
    }

    #[test]
    fn prefill_abab_mediatek_ffn_selects_and_updates_only_mediatek_env() {
        let mode = prefill_abab_mode_from_lookup(|var| {
            (var == "RNB_PREFILL_ABAB_MEDIATEK_FFN").then(|| "1".to_string())
        })
        .expect("MediaTek ABAB mode should parse");

        assert_eq!(mode, Some(PrefillAbabMode::MediatekFfn));
        assert_eq!(
            prefill_abab_env_updates(mode, true),
            vec![("RNB_MEDIATEK_PREFILL_FFN", "1")]
        );
        assert_eq!(
            prefill_abab_env_updates(mode, false),
            vec![("RNB_MEDIATEK_PREFILL_FFN", "0")]
        );
    }

    #[test]
    fn prefill_abab_atn_full_layer_updates_only_atn_envs() {
        let updates = prefill_abab_env_updates(Some(PrefillAbabMode::AtnFullLayer), true);

        assert_eq!(
            updates,
            vec![
                ("RNB_METAL_PREFILL_ATN_FULL_LAYER", "1"),
                ("RNB_METAL_PREFILL_ATN_FULL_TIME", "1"),
            ]
        );
        assert!(updates
            .iter()
            .all(|(var, _)| !var.contains("GDN") && !var.contains("FLASH")));
    }

    #[test]
    fn prefill_abab_atn_full_layer_has_labels_and_counter_contract() {
        let mode = Some(PrefillAbabMode::AtnFullLayer);

        assert_eq!(
            prefill_abab_labels(mode),
            ("atn full/core ON", "atn full/core OFF")
        );
        assert!(!prefill_abab_label_scoped_argmax(mode));
        assert!(prefill_abab_reports_atn_counters(mode));
        assert!(!prefill_abab_reports_atn_counters(Some(
            PrefillAbabMode::AttnChain
        )));
    }

    #[test]
    fn prefill_abab_atn_o_tail_updates_only_o_tail_envs() {
        let updates = prefill_abab_env_updates(Some(PrefillAbabMode::AtnOTail), true);

        assert_eq!(
            updates,
            vec![
                ("RNB_METAL_PREFILL_ATN_O_TAIL", "1"),
                ("RNB_METAL_PREFILL_ATN_O_TAIL_TIME", "1"),
            ]
        );
        assert!(updates
            .iter()
            .all(|(var, _)| !var.contains("GDN") && !var.contains("FLASH")));
    }

    #[test]
    fn prefill_abab_atn_o_tail_has_labels_and_counter_contract() {
        let mode = Some(PrefillAbabMode::AtnOTail);

        assert_eq!(
            prefill_abab_labels(mode),
            ("atn o-tail ON", "atn o-tail OFF")
        );
        assert!(!prefill_abab_label_scoped_argmax(mode));
        assert!(prefill_abab_reports_atn_counters(mode));
    }

    #[test]
    fn load_prompt_text_truncates_file_on_utf8_boundaries() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "rnb-llm-bench-prompt-{}-{}.txt",
            std::process::id(),
            "utf8"
        ));
        std::fs::write(&path, "대한민국의 수도는").expect("write utf8 prompt");
        let prev_prompt = std::env::var("RNB_PROMPT").ok();
        let prev_file = std::env::var("RNB_PROMPT_FILE").ok();
        std::env::remove_var("RNB_PROMPT");
        std::env::set_var("RNB_PROMPT_FILE", &path);

        let text = load_prompt_text(1, "fallback");

        assert_eq!(text, "대한민국의 수도");
        std::fs::remove_file(&path).expect("remove utf8 prompt");
        if let Some(prev) = prev_prompt {
            std::env::set_var("RNB_PROMPT", prev);
        } else {
            std::env::remove_var("RNB_PROMPT");
        }
        if let Some(prev) = prev_file {
            std::env::set_var("RNB_PROMPT_FILE", prev);
        } else {
            std::env::remove_var("RNB_PROMPT_FILE");
        }
    }

    #[test]
    fn test_nemotron_thinking_default_can_be_overridden_by_prompt() {
        assert!(nemotron_thinking_enabled_with_default("solve this", true));
        assert!(!nemotron_thinking_enabled_with_default("solve this", false));
        assert!(nemotron_thinking_enabled_with_default(
            "/think solve this",
            false
        ));
        assert!(!nemotron_thinking_enabled_with_default(
            "/no_think solve this",
            true
        ));
    }

    #[test]
    fn test_nemotron_chat_prompt_renders_reasoning_prefix() {
        let rendered = build_nemotron_chat_prompt_text_with_options(
            "What is 84 * 3 / 2?",
            "You are concise.",
            true,
        );
        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are concise.<|im_end|>\n\
<|im_start|>user\nWhat is 84 * 3 / 2?<|im_end|>\n\
<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn test_nemotron_chat_prompt_can_disable_reasoning() {
        let rendered =
            build_nemotron_chat_prompt_text_with_options("/no_think Give final only.", "", false);
        assert_eq!(
            rendered,
            "<|im_start|>system\n<|im_end|>\n\
<|im_start|>user\nGive final only.<|im_end|>\n\
<|im_start|>assistant\n<think></think>"
        );
    }

    #[test]
    fn cuda_argmax_auto_is_disabled_for_speculative_modes() {
        assert!(!cuda_exact_greedy_auto_allowed_for_backend(
            Some("1"),
            false,
            false,
            1.0,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
        ));
        assert!(!cuda_exact_greedy_auto_allowed_for_backend(
            Some("2"),
            false,
            false,
            1.0,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
        ));
    }

    #[test]
    fn cuda_argmax_auto_is_allowed_for_plain_greedy_decode() {
        assert!(cuda_exact_greedy_auto_allowed_for_backend(
            None, false, false, 1.0, false, false, false, false, false, false, false, true,
        ));
    }

    #[test]
    fn metal_argmax_auto_is_allowed_for_plain_greedy_decode() {
        assert!(metal_exact_greedy_auto_allowed_for_backend(
            None, false, false, 1.0, false, false, false, false, false, false, false, true,
        ));
    }

    #[test]
    fn cuda_argmax_auto_is_disabled_when_backend_argmax_is_unsupported() {
        assert!(!cuda_exact_greedy_auto_allowed_for_backend(
            None, false, false, 1.0, false, false, false, false, false, false, false, false,
        ));
    }

    #[test]
    fn standard_generate_mode_is_only_for_non_spec_runs() {
        assert!(standard_generate_mode_requested(None, true));
        assert!(!standard_generate_mode_requested(Some("1"), true));
        assert!(!standard_generate_mode_requested(None, false));
    }

    #[test]
    fn mtp_generate_mode_does_not_require_rnb_generate() {
        assert!(mtp_generate_mode_requested(None, true, false));
    }

    #[test]
    fn mtp_generate_mode_overrides_legacy_spec_one() {
        assert!(mtp_generate_mode_requested(Some("1"), true, false));
    }

    #[test]
    fn mtp_generate_mode_does_not_override_two_model_spec() {
        assert!(!mtp_generate_mode_requested(Some("2"), true, false));
    }

    #[test]
    fn mtp_generate_mode_keeps_explicit_generate_path() {
        assert!(mtp_generate_mode_requested(None, false, true));
    }

    #[test]
    fn mtp_env_request_distinguishes_auto_from_force() {
        assert_eq!(parse_mtp_env_request(None), MtpEnvRequest::Off);
        assert_eq!(parse_mtp_env_request(Some("0")), MtpEnvRequest::Off);
        assert_eq!(parse_mtp_env_request(Some("false")), MtpEnvRequest::Off);
        assert_eq!(parse_mtp_env_request(Some("auto")), MtpEnvRequest::Auto);
        assert_eq!(parse_mtp_env_request(Some("1")), MtpEnvRequest::Force);
    }

    #[test]
    fn generated_text_sha256_is_stable_for_sequence_gate_logs() {
        assert_eq!(
            generated_text_sha256_hex("Gemma4 MTP"),
            "902c87300f29b58b597d8c38fc8521a150cdbde648fb8e62aecf2115ee50c65f"
        );
    }

    #[test]
    fn mtp_auto_request_only_generates_when_policy_is_enabled() {
        let disabled = mtp_test_policy(false, 4, true);
        let enabled = mtp_test_policy(true, 1, true);

        assert!(!mtp_env_requests_generation(MtpEnvRequest::Auto, disabled));
        assert!(mtp_env_requests_generation(MtpEnvRequest::Auto, enabled));
        assert!(mtp_env_requests_generation(MtpEnvRequest::Force, disabled));
    }

    fn mtp_test_policy(
        enabled: bool,
        spec_k: usize,
        device_verify: bool,
    ) -> rnb_llm::engine::MtpAutoPolicy {
        rnb_llm::engine::MtpAutoPolicy {
            enabled,
            spec_k,
            device_verify,
            min_free_vram_mib: 2048,
            resource: Some(rnb_llm::engine::MtpAutoResourceHint {
                total_vram_mib: 12 * 1024,
                free_vram_mib: 10 * 1024,
            }),
            reason: "test",
        }
    }

    #[test]
    fn mtp_auto_enables_device_verify_when_policy_requests_it() {
        assert!(mtp_should_enable_device_verify(
            MtpEnvRequest::Auto,
            mtp_test_policy(true, 1, true),
            false,
        ));
        assert!(!mtp_should_enable_device_verify(
            MtpEnvRequest::Auto,
            mtp_test_policy(false, 1, true),
            false,
        ));
        assert!(!mtp_should_enable_device_verify(
            MtpEnvRequest::Auto,
            mtp_test_policy(true, 1, true),
            true,
        ));
    }

    #[test]
    fn mtp_force_uses_policy_device_verify_when_env_is_unset() {
        assert!(mtp_should_enable_device_verify(
            MtpEnvRequest::Force,
            mtp_test_policy(false, 4, true),
            false,
        ));
    }

    #[test]
    fn mtp_spec_k_uses_policy_when_unset_or_auto() {
        let policy = mtp_test_policy(true, 1, true);

        assert_eq!(resolve_mtp_spec_k(None, policy).unwrap(), 1);
        assert_eq!(resolve_mtp_spec_k(Some("auto"), policy).unwrap(), 1);
        assert_eq!(resolve_mtp_spec_k(Some("3"), policy).unwrap(), 3);
    }

    #[test]
    fn mtp_spec_k_rejects_zero_and_invalid_values() {
        let policy = mtp_test_policy(true, 1, true);

        assert!(resolve_mtp_spec_k(Some("0"), policy).is_err());
        assert!(resolve_mtp_spec_k(Some("bad"), policy).is_err());
    }

    #[test]
    fn mtp_abab_repeat_parser_accepts_even_run_counts() {
        assert_eq!(parse_mtp_abab_repeat(None).unwrap(), None);
        assert_eq!(parse_mtp_abab_repeat(Some("2")).unwrap(), Some(2));
        assert_eq!(parse_mtp_abab_repeat(Some(" 8 ")).unwrap(), Some(8));
    }

    #[test]
    fn mtp_abab_spec_k_b_parser_accepts_positive_values() {
        assert_eq!(parse_mtp_abab_spec_k_b(None).unwrap(), None);
        assert_eq!(parse_mtp_abab_spec_k_b(Some(" 2 ")).unwrap(), Some(2));
    }

    #[test]
    fn mtp_abab_spec_k_b_parser_rejects_zero_and_invalid_values() {
        for raw in ["", "0", "bad"] {
            assert!(parse_mtp_abab_spec_k_b(Some(raw)).is_err(), "{raw:?}");
        }
    }

    #[test]
    fn mtp_abab_rejects_profile_and_trace_io_inside_timer() {
        assert_eq!(validate_mtp_abab_timing_env(false, false, false), Ok(()));
        for polluted in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
        ] {
            assert!(
                validate_mtp_abab_timing_env(polluted.0, polluted.1, polluted.2)
                    .unwrap_err()
                    .contains("timer-I/O-free")
            );
        }
    }

    #[test]
    fn mtp_abab_repeat_parser_rejects_small_odd_and_invalid_values() {
        for raw in ["", "0", "1", "3", "bad"] {
            assert!(parse_mtp_abab_repeat(Some(raw)).is_err(), "{raw:?}");
        }
    }

    #[test]
    fn mtp_abab_order_alternates_sequential_and_batch_prefill() {
        let variants = (0..8).map(MtpAbabVariant::for_run).collect::<Vec<_>>();
        assert_eq!(
            variants,
            vec![
                MtpAbabVariant::Sequential,
                MtpAbabVariant::BatchPrefill,
                MtpAbabVariant::Sequential,
                MtpAbabVariant::BatchPrefill,
                MtpAbabVariant::Sequential,
                MtpAbabVariant::BatchPrefill,
                MtpAbabVariant::Sequential,
                MtpAbabVariant::BatchPrefill,
            ]
        );
    }

    #[test]
    fn mtp_abab_median_excludes_only_cold_a1() {
        let (a_median, b_median) =
            mtp_abab_medians(&[1_000.0, 30.0, 10.0, 20.0], &[40.0, 20.0, 30.0, 10.0]);
        assert_eq!(a_median, Some(20.0));
        assert_eq!(b_median, Some(25.0));
        assert_eq!(mtp_abab_medians(&[1_000.0], &[40.0]), (None, Some(40.0)));
    }

    #[test]
    fn mtp_abab_result_gate_uses_first_run_for_cross_variant_exactness() {
        assert_eq!(mtp_abab_result_equality(None, "canonical", 3), None);
        let canonical = ("canonical".to_string(), 3);
        assert_eq!(
            mtp_abab_result_equality(Some(&canonical), "canonical", 3),
            Some(true)
        );
        assert_eq!(
            mtp_abab_result_equality(Some(&canonical), "variant-b", 3),
            Some(false)
        );
        assert_eq!(
            mtp_abab_result_equality(Some(&canonical), "canonical", 4),
            Some(false)
        );
    }

    #[test]
    fn mtp_abab_timed_callback_is_noop_continuation() {
        assert!(mtp_abab_continue(""));
        assert!(mtp_abab_continue("generated piece"));
    }

    #[test]
    fn mtp_abab_draft_only_gate_matches_runtime_truthy_values() {
        for raw in ["1", "true", "on", "yes", "unexpected", ""] {
            assert!(mtp_abab_draft_only_requested(Some(raw)), "{raw:?}");
        }
        for raw in ["0", "false", "off", "no"] {
            assert!(!mtp_abab_draft_only_requested(Some(raw)), "{raw:?}");
        }
        assert!(!mtp_abab_draft_only_requested(None));
    }

    #[test]
    fn mtp_abab_rejects_paths_that_do_not_consume_batch_verify_toggle() {
        let valid = validate_mtp_abab_path(
            true,
            None,
            false,
            true,
            true,
            rnb_loader::Architecture::Qwen35,
        );
        assert_eq!(valid, Ok(()));

        let standard = validate_mtp_abab_path(
            false,
            None,
            false,
            true,
            true,
            rnb_loader::Architecture::Qwen35,
        );
        assert!(standard.unwrap_err().contains("in-model MTP"));

        let speculative = validate_mtp_abab_path(
            true,
            Some("2"),
            false,
            true,
            true,
            rnb_loader::Architecture::Qwen35,
        );
        assert!(speculative.unwrap_err().contains("RNB_SPEC=2"));

        let draft_only = validate_mtp_abab_path(
            true,
            None,
            true,
            true,
            true,
            rnb_loader::Architecture::Qwen35,
        );
        assert!(draft_only.unwrap_err().contains("RNB_MTP_DRAFT_ONLY"));

        let external = validate_mtp_abab_path(
            true,
            None,
            false,
            true,
            true,
            rnb_loader::Architecture::Gemma4,
        );
        assert!(external.unwrap_err().contains("external drafter"));

        let no_weights = validate_mtp_abab_path(
            true,
            None,
            false,
            false,
            true,
            rnb_loader::Architecture::Qwen35,
        );
        assert!(no_weights.unwrap_err().contains("model weights"));

        let no_mtp = validate_mtp_abab_path(
            true,
            None,
            false,
            true,
            false,
            rnb_loader::Architecture::Qwen35,
        );
        assert!(no_mtp.unwrap_err().contains("MTP runtime"));
    }

    #[test]
    fn mtp_abab_env_restore_restores_after_normal_completion() {
        const KEY: &str = "RNB_TEST_MTP_ABAB_ENV_RESTORE";
        let restore_original = EnvVarRestore::capture(KEY);
        // SAFETY: 이 테스트 전용 env 이름은 다른 코드에서 읽지 않는다.
        unsafe {
            std::env::set_var(KEY, "before");
        }
        {
            let _restore_before = EnvVarRestore::set_temporarily(KEY, "during");
            assert_eq!(std::env::var(KEY).as_deref(), Ok("during"));
        }
        assert_eq!(std::env::var(KEY).as_deref(), Ok("before"));
        drop(restore_original);
    }

    #[test]
    fn mtp_abab_env_restore_restores_after_error_return() {
        const KEY: &str = "RNB_TEST_MTP_ABAB_ENV_RESTORE_ERROR";
        let restore_original = EnvVarRestore::capture(KEY);
        // SAFETY: 이 테스트 전용 env 이름은 다른 코드에서 읽지 않는다.
        unsafe {
            std::env::set_var(KEY, "before");
        }

        let result: Result<(), &'static str> = (|| {
            let _restore_before = EnvVarRestore::capture(KEY);
            // SAFETY: 이 테스트 전용 env 이름은 다른 코드에서 읽지 않는다.
            unsafe {
                std::env::set_var(KEY, "during");
            }
            Err("expected")
        })();

        assert_eq!(result, Err("expected"));
        assert_eq!(std::env::var(KEY).as_deref(), Ok("before"));
        drop(restore_original);
    }

    #[test]
    fn mtp_abab_env_restore_restores_after_panic() {
        const KEY: &str = "RNB_TEST_MTP_ABAB_ENV_RESTORE_PANIC";
        let restore_original = EnvVarRestore::capture(KEY);
        // SAFETY: 이 테스트 전용 env 이름은 다른 코드에서 읽지 않는다.
        unsafe {
            std::env::set_var(KEY, "before");
        }

        let result = std::panic::catch_unwind(|| {
            let _restore_before = EnvVarRestore::capture(KEY);
            // SAFETY: 이 테스트 전용 env 이름은 다른 코드에서 읽지 않는다.
            unsafe {
                std::env::set_var(KEY, "during");
            }
            panic!("expected");
        });

        assert!(result.is_err());
        assert_eq!(std::env::var(KEY).as_deref(), Ok("before"));
        drop(restore_original);
    }

    #[test]
    fn quiet_decode_suppresses_spec_piece_output() {
        assert!(!decode_piece_output_enabled(true));
        assert!(decode_piece_output_enabled(false));
    }

    #[test]
    fn verify_microbench_config_clamps_zero_values() {
        assert_eq!(
            parse_verify_microbench_config(Some("0"), Some("0")),
            VerifyMicrobenchConfig {
                window: 1,
                rounds: 1,
            }
        );
    }

    #[test]
    fn verify_microbench_config_uses_positive_values() {
        assert_eq!(
            parse_verify_microbench_config(Some("4"), Some("3")),
            VerifyMicrobenchConfig {
                window: 4,
                rounds: 3,
            }
        );
    }

    #[cfg(feature = "mediatek")]
    #[test]
    fn mediatek_quant_probe_shape_parser_accepts_default_and_explicit_shape() {
        assert_eq!(
            parse_mtk_quant_gated_gelu_probe_shape(None).unwrap(),
            rnb_backend_mediatek::MediaTekGatedGeluFfnShape::new(1536, 6144, 1536)
        );
        assert_eq!(
            parse_mtk_quant_gated_gelu_probe_shape(Some("128,512,256")).unwrap(),
            rnb_backend_mediatek::MediaTekGatedGeluFfnShape::new(128, 512, 256)
        );
    }

    #[cfg(feature = "mediatek")]
    #[test]
    fn mediatek_quant_probe_shape_parser_rejects_bad_shape() {
        assert!(parse_mtk_quant_gated_gelu_probe_shape(Some("128,512")).is_err());
        assert!(parse_mtk_quant_gated_gelu_probe_shape(Some("128,0,256")).is_err());
        assert!(parse_mtk_quant_gated_gelu_probe_shape(Some("bad,512,256")).is_err());
    }

    #[cfg(feature = "mediatek")]
    #[test]
    fn mediatek_quant_probe_result_format_is_machine_readable() {
        let result = rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnSupportResult::new(
            rnb_backend_mediatek::MediaTekNnapiDeviceInfo::new(
                "mtk-neuron_shim",
                rnb_backend_mediatek::MEDIATEK_NNAPI_DEVICE_ACCELERATOR,
                1000008,
                "7.2.4",
            ),
            rnb_backend_mediatek::MediaTekGatedGeluFfnSupportedOps::all(true),
            11,
            22,
        );

        assert_eq!(
            format_mtk_quant_gated_gelu_probe_result(
                rnb_backend_mediatek::MediaTekGatedGeluFfnShape::new(1536, 6144, 1536),
                &result,
            ),
            "[mediatek-quant-ffn-probe] shape=1536x6144x1536 device=mtk-neuron_shim device_type=4 feature_level=1000008 supported=true supported_ops=gate_fc=true,up_fc=true,gelu_square_mul=true,gelu_cube_mul=true,gelu_poly_scale_mul=true,gelu_poly_add=true,gelu_tanh_scale_mul=true,gelu_tanh=true,gelu_one_plus_add=true,gelu_gate_one_plus_mul=true,gelu_half_mul=true,gated_mul=true,down_fc=true model_build_ns=11 supported_ops_query_ns=22"
        );
    }
}
