use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::backend_runtime::{
    init_engine_backend_runtime, reset_backend_state_for_engine_init, EngineBackendRuntime,
};
use super::layer_weights::LayerType;
use super::layout::resolve_attention_layout;
use super::load_profile::LoadProfile;
use super::model_init::{build_model_metadata, build_tokenizer};
use super::models::shared_expert_moe::wire_sparse_expert_page_cache;
use super::moe_section::{attach_moe_section_decode, moe_section_decode_sidecar_requested};
use super::mtp::{EngineMtpRuntime, EngineMtpState, InModelMtpRuntime};
use super::packed_wiring::wire_shadow_model;
use super::packed_wiring::{
    open_diagnostic_packed_model, open_shadow_model, wire_packed_model_weights,
};
use super::state::Engine;
use super::threading::configure_cpu_runtime;
use super::types::{ModelMetadata, ScratchBuffers};
use super::weight_loading::{load_model_weights, load_mtp_layer_weights};
use crate::kv_cache::KVCache;
use crate::tokenizer::Tokenizer;
use rnb_loader::Architecture as ModelArchitecture;

fn maybe_enable_q4k_prefill_f16_gemm_for_dense_attention(_all_layers_attention: bool) {}

#[cfg(test)]
pub(in crate::engine) fn maybe_enable_q4k_prefill_f16_gemm_for_test(all_layers_attention: bool) {
    maybe_enable_q4k_prefill_f16_gemm_for_dense_attention(all_layers_attention);
}

#[cfg(feature = "cuda")]
fn detected_cuda_memory_bytes() -> Option<(u64, u64)> {
    crate::engine::cuda_runtime::cuda_memory_info()
        .ok()
        .map(|info| (info.free_bytes as u64, info.total_bytes as u64))
}

#[cfg(not(feature = "cuda"))]
fn detected_cuda_memory_bytes() -> Option<(u64, u64)> {
    None
}
fn apply_host_memory_plan(
    weights: &mut super::layer_weights::ModelWeights,
    sparse_moe_cuda_enabled: bool,
) {
    for layer in &mut weights.layers {
        let moe = match layer {
            LayerType::Attention(weights) => weights.shared_expert_moe.as_mut(),
            LayerType::GatedDeltaNet(weights) => weights.shared_expert_moe.as_mut(),
            LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => None,
        };
        if let Some(moe) = moe {
            moe.prefer_sparse_moe_cuda = sparse_moe_cuda_enabled;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EngineLoadConfig {
    pub diagnostic_sidecar: Option<PathBuf>,
    pub host_memory_budget: Option<rnb_runtime::memory::MemoryBudget>,
}

impl EngineLoadConfig {
    pub fn with_diagnostic_sidecar(mut self, sidecar: Option<PathBuf>) -> Self {
        self.diagnostic_sidecar = sidecar;
        self
    }

    pub fn with_host_ram_budget_bytes(mut self, bytes: u64) -> Self {
        self.host_memory_budget = (bytes > 0).then(|| {
            rnb_runtime::memory::MemoryBudget::new(rnb_runtime::memory::MemoryTier::Ram, bytes, 0)
        });
        self
    }
}

impl Engine {
    pub fn from_gguf(path: &Path) -> crate::error::Result<Self> {
        Self::from_gguf_with_config(path, EngineLoadConfig::default())
    }
    pub fn from_gguf_with_host_ram_budget(path: &Path, bytes: u64) -> crate::error::Result<Self> {
        Self::from_gguf_with_config(
            path,
            EngineLoadConfig::default().with_host_ram_budget_bytes(bytes),
        )
    }

    /// Loads a GGUF model with runtime resource policy.
    ///
    /// `host_memory_budget` overrides the physical-RAM-derived automatic budget
    /// for engine-owned host residency and cache choices. GGUF mappings are
    /// file-backed and remain reclaimable by the OS, so this is not an
    /// operating-system RSS hard limit.
    pub fn from_gguf_with_config(
        path: &Path,
        config: EngineLoadConfig,
    ) -> crate::error::Result<Self> {
        let mut load_profile = LoadProfile::from_env();
        let load_profile_total_start = load_profile.begin();
        macro_rules! load_stage {
            ($name:literal, $body:block) => {{
                let stage_start = load_profile.begin();
                let value = $body;
                load_profile.record_since($name, stage_start);
                value
            }};
        }

        load_stage!("reset_backend_state", {
            reset_backend_state_for_engine_init()
        })?;
        let EngineLoadConfig {
            diagnostic_sidecar,
            host_memory_budget,
        } = config;

        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("rnb"))
        {
            return Err(crate::error::LlmError::ModelLoad(
                "standalone .rnb files are no longer supported; pass the GGUF model file instead"
                    .into(),
            ));
        }

        load_stage!("thread_policy_probe", {
            let moe_section_decode_sidecar = moe_section_decode_sidecar_requested(path);
            let architecture_for_thread_policy = rnb_loader::detect_model_architecture(path).ok();
            configure_cpu_runtime(
                path,
                moe_section_decode_sidecar,
                architecture_for_thread_policy,
            );
        });

        let model = load_stage!("load_model", { rnb_loader::load_model(path) })
            .map_err(|e| crate::error::LlmError::ModelLoad(e.to_string()))?;
        let diagnostic_sidecar = diagnostic_sidecar.as_deref();

        // Sum mapped tensor payloads rather than the selected path's file
        // length. A split GGUF's first shard may be a tiny metadata/tensor
        // index file while the remaining shards hold hundreds of GiB.
        let gguf_mapped_weight_bytes = model.weights.values().fold(0u64, |total, tensor| {
            total.saturating_add(tensor.as_bytes().map_or(0, |bytes| bytes.len() as u64))
        });
        let mapped_weight_bytes = gguf_mapped_weight_bytes.saturating_add(
            diagnostic_sidecar
                .and_then(|sidecar| std::fs::metadata(sidecar).ok())
                .map(|metadata| metadata.len())
                .unwrap_or_default(),
        );
        let cuda_memory_bytes = detected_cuda_memory_bytes();
        let host_memory_plan = rnb_runtime::policy::HostMemoryPlan::automatic(
            host_memory_budget,
            mapped_weight_bytes,
            cuda_memory_bytes.is_some(),
        );
        let sparse_moe_cuda_enabled = rnb_runtime::policy::cuda_q2k_q3k_sparse_moe_enabled(
            host_memory_plan.prefer_sparse_moe_cuda(),
        );
        if let Some(budget) = host_memory_plan.ram_budget() {
            eprintln!(
                "[INFO] Host RAM budget: {:.2} GiB, total: {:.2} GiB, source={}, mapped weights: {:.2} GiB, constrained={}, sparse_moe_cuda={}",
                budget.available_bytes() as f64 / (1024_u64.pow(3)) as f64,
                budget.total_bytes() as f64 / (1024_u64.pow(3)) as f64,
                if host_memory_plan.uses_automatic_budget() {
                    "automatic"
                } else {
                    "application"
                },
                host_memory_plan.mapped_weight_bytes() as f64 / (1024_u64.pow(3)) as f64,
                host_memory_plan.is_constrained(),
                sparse_moe_cuda_enabled,
            );
        }
        if let Some((free_bytes, total_bytes)) = cuda_memory_bytes {
            eprintln!(
                "[INFO] CUDA VRAM: {:.2} GiB free / {:.2} GiB total",
                free_bytes as f64 / (1024_u64.pow(3)) as f64,
                total_bytes as f64 / (1024_u64.pow(3)) as f64,
            );
        }

        // Weight 매핑 검증
        let weight_count = model.weights.len();
        eprintln!("[INFO] GGUF weights: {}", weight_count);

        let (tokenizer, vocab_size) = load_stage!("build_tokenizer", { build_tokenizer(&model) });
        let metadata = load_stage!("build_metadata", {
            build_model_metadata(&model, vocab_size)
        });
        let mtp = load_stage!("build_mtp_state", {
            EngineMtpState::from_loader_parts(model.metadata.mtp.as_ref(), &model.mtp_tensors)
        })?;
        if let Some(mtp) = &mtp {
            eprintln!(
                "[INFO] MTP head: trunk_layers={}, nextn_layers={}, first_mtp_layer={}",
                mtp.trunk_layers, mtp.nextn_predict_layers, mtp.first_mtp_layer
            );
        }

        // Product loads stay on GGUF direct. Packed RNBC data is considered only
        // when a diagnostic caller supplies an explicit path.
        let packed_model = load_stage!("open_packed_model", {
            open_diagnostic_packed_model(diagnostic_sidecar)
        });

        // Pre-dequantize all weights
        let mut weights = load_stage!("load_model_weights", {
            load_model_weights(
                &model,
                packed_model.as_deref(),
                metadata.num_layers,
                metadata.full_attention_interval,
                path,
            )
        });
        apply_host_memory_plan(&mut weights, sparse_moe_cuda_enabled);
        let mtp_runtime = load_stage!("load_mtp_weights", {
            mtp.as_ref().and_then(|_| {
                load_mtp_layer_weights(&model).map(|mtp_weights| {
                    eprintln!(
                        "[INFO] MTP runtime loaded: layer={} eh_proj={:?} [{}x{}]",
                        mtp_weights.layer_index,
                        mtp_weights.eh_proj.ggml_type,
                        mtp_weights.eh_proj.rows,
                        mtp_weights.eh_proj.cols
                    );
                    EngineMtpRuntime::InModel(InModelMtpRuntime::new(&metadata, mtp_weights))
                })
            })
        });

        let (layer_num_kv_heads, layer_head_dims): (Vec<usize>, Vec<usize>) =
            load_stage!("plan_layer_cache_shapes", {
                let layer_num_kv_heads: Vec<usize> = weights
                    .layers
                    .iter()
                    .enumerate()
                    .map(|(layer_idx, layer)| match layer {
                        LayerType::Attention(_) | LayerType::GatedDeltaNet(_) => metadata
                            .head_count_kv_per_layer
                            .as_ref()
                            .and_then(|v| v.get(layer_idx).copied())
                            .unwrap_or(metadata.num_kv_heads),
                        LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => 0,
                    })
                    .collect();
                let layer_head_dims: Vec<usize> = weights
                    .layers
                    .iter()
                    .enumerate()
                    .map(|(layer_idx, layer)| match layer {
                        LayerType::Attention(w) => {
                            let kv_override = metadata
                                .head_count_kv_per_layer
                                .as_ref()
                                .and_then(|v| v.get(layer_idx).copied());
                            resolve_attention_layout(&metadata, w, kv_override)
                                .map(|l| l.head_dim)
                                .unwrap_or(metadata.head_dim)
                        }
                        LayerType::GatedDeltaNet(_) => metadata.head_dim,
                        LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => {
                            metadata.head_dim
                        }
                    })
                    .collect();
                (layer_num_kv_heads, layer_head_dims)
            });

        let mut kv_cache = load_stage!("kv_cache_alloc", {
            KVCache::new_per_layer(metadata.max_seq_len, &layer_num_kv_heads, &layer_head_dims)
        });

        // SSM state 초기화 (GDN/Mamba2 레이어용)
        load_stage!("ssm_state_init", {
            if weights.layers.iter().any(|layer| {
                matches!(
                    layer,
                    LayerType::GatedDeltaNet(_) | LayerType::NemotronMamba2(_)
                )
            }) {
                let d_inner = metadata.ssm_d_inner;
                let d_state = metadata.ssm_d_state;
                let n_group = metadata.ssm_n_group;
                let dt_rank = metadata.ssm_dt_rank;
                let conv_kernel = metadata.ssm_conv_kernel;
                let conv_channels = d_inner + 2 * n_group * d_state;
                let head_v_dim = d_inner / dt_rank.max(1);
                let head_k_dim = d_state;
                let num_heads = dt_rank;

                for i in 0..metadata.num_layers {
                    if matches!(
                        weights.layers.get(i),
                        Some(LayerType::GatedDeltaNet(_) | LayerType::NemotronMamba2(_))
                    ) {
                        kv_cache.init_ssm_state(
                            i,
                            conv_kernel,
                            conv_channels,
                            num_heads,
                            head_v_dim,
                            head_k_dim,
                        );
                    }
                }
                eprintln!(
                    "[INFO] SSM state initialized: conv=[{}, {}], delta=[{}, {}, {}]",
                    conv_kernel - 1,
                    conv_channels,
                    num_heads,
                    head_v_dim,
                    head_k_dim
                );
            }
        });

        eprintln!(
            "[INFO] Model: hidden={}, heads={}, kv_heads={}, head_dim={}",
            metadata.hidden_dim, metadata.num_heads, metadata.num_kv_heads, metadata.head_dim
        );
        if metadata.rope_dim > 0 {
            eprintln!(
                "[INFO] RoPE: partial dim={}/{}, sections={:?}, theta={}",
                metadata.rope_dim, metadata.head_dim, metadata.rope_sections, metadata.rope_theta
            );
        }

        // Determine FFN inner dim from the maximum gate rows across all layers.
        // Gemma E2B uses per-layer FFN lengths (e.g. 6144 / 12288), so first-layer sizing is unsafe.
        let ffn_inner_dim = weights
            .layers
            .iter()
            .map(|l| match l {
                LayerType::Attention(w) => w.ffn_gate_weight.rows,
                LayerType::GatedDeltaNet(w) => w.ffn_gate_weight.rows,
                LayerType::NemotronMamba2(_) => 0,
                LayerType::NemotronMoE(w) => w.expert_up.rows,
            })
            .max()
            .unwrap_or(0);

        let scratch = load_stage!("scratch_alloc", {
            ScratchBuffers::new(&metadata, ffn_inner_dim)
        });

        // Mixed-precision shadow weights are diagnostic-only. The runtime opens
        // `<stem>.shadow.rnb` only when `RNB_SHADOW_WEIGHTS=1` is explicit.
        let shadow_model = load_stage!("open_shadow_model", { open_shadow_model(path) });

        // packed_model에서 각 QuantizedWeight에 packed_gemm_data 연결
        load_stage!("wire_packed_model", {
            if let Some(ref pm) = packed_model {
                wire_packed_model_weights(
                    path,
                    &mut weights,
                    &metadata,
                    model.metadata.architecture,
                    pm,
                );
            }
        });

        // Wire an explicitly requested shadow model independently of the RNBC
        // diagnostic path, including when `RNB_FORCE_GGUF=1`.
        // Session 73 Task 8: arch 분기 추가 — Qwen35MoE 는 gate/up/down(optional) 개별 shadow,
        // 기존 Gemma4 경로는 `_ =>` arm 에 원문 그대로 보존.
        load_stage!("wire_shadow_model", {
            if let Some(ref sm) = shadow_model {
                wire_shadow_model(&mut weights, model.metadata.architecture, sm);
            }
        });
        if let Some(budget) = host_memory_plan.ram_budget() {
            if budget.available_bytes() < gguf_mapped_weight_bytes {
                if let Some(expert_budget_bytes) = wire_sparse_expert_page_cache(
                    &mut weights,
                    budget.available_bytes(),
                    gguf_mapped_weight_bytes,
                ) {
                    eprintln!(
                        "[INFO] Sparse expert page-cache budget: {:.2} GiB",
                        expert_budget_bytes as f64 / (1024_u64.pow(3)) as f64,
                    );
                }
            }
        }

        // Legacy RNBM MoE sections are diagnostic-only. `RNB_MOE_DECODE=1`
        // explicitly enables sibling `<model>.rnb` detection; the unset product
        // default stays on GGUF weights.
        let moe_section_decode_bytes: Option<Arc<memmap2::Mmap>> =
            load_stage!("attach_moe_section", {
                attach_moe_section_decode(path, &mut weights)
            });

        let all_layers_attention = !weights.layers.is_empty()
            && weights
                .layers
                .iter()
                .all(|l| matches!(l, LayerType::Attention(_)));
        maybe_enable_q4k_prefill_f16_gemm_for_dense_attention(all_layers_attention);

        // cu35→cu36 revert: Qwen3.6 ABAB 8-pair (cu36) 일관 회귀 (-4.2%
        // prefill but +5.7% decode, total +3.3%). cu34 단일 run -20% 측정
        // wrong (variance). auto-set 제거. 사용자가 RNB_CUDA_MOE_LAYER_CACHE=1
        // 또는 RNB_CUDA_MOE_LAYER_CACHE_MB=N 명시 set 으로 opt-in 가능.
        // cu37+ Mamba SSM kernel / Gemma E4B 추가 진단으로 진짜 lever 탐색.

        let backend_runtime = load_stage!("backend_runtime_init", {
            init_engine_backend_runtime(&metadata, &weights, ffn_inner_dim)
        });
        load_stage!("backend_register_moe", {
            register_moe_layers_with_backend(&weights, &metadata)
        })?;
        load_stage!("backend_prewarm_q4_gate_up", {
            super::backend_runtime::prewarm_dense_q4_packed_gate_up_weights(&weights)
        })?;
        load_stage!("backend_prewarm_q6_down", {
            super::backend_runtime::prewarm_dense_q6_packed_down_weights(&weights)
        })?;
        load_stage!("backend_prewarm_q4_prefill", {
            super::backend_runtime::prewarm_prefill_q4_f32_projection_weights(&weights)
        })?;
        // cu19: a broader Q4_K raw-weights prewarm was tried but measured
        // ~0 effect on Gemma 4 prefill — when the cache fills it triggers LRU
        // eviction during the first prefill, so the early layers are evicted
        // before forward visits them. The launch_q4k_dequant_* path's
        // cache-miss branch already handles the H2D + insert. Re-introducing
        // a full prewarm makes init slower and on tight-cache devices
        // (cache_limit < total dense Q4_K bytes) thrashes without benefit.

        let mut engine = load_stage!("engine_construct", {
            Self {
                tokenizer,
                kv_cache,
                metadata,
                architecture: model.metadata.architecture,
                host_memory_plan,
                weights: Some(weights),
                scratch: Some(scratch),
                mtp,
                mtp_runtime,
                backend_runtime,
                packed_model,
                shadow_model,
                memtrace_step: std::sync::atomic::AtomicUsize::new(0),
                moe_section_decode_bytes,
                #[cfg(feature = "vulkan")]
                fullpath_token_embd_bound: false,
                last_layer_hidden_cached: Vec::new(),
            }
        });
        load_stage!("axis_p_mlock", {
            engine.apply_axis_p_mlock();
        });

        // mc78 Phase 1: Auto-detect sibling external drafter. Only fires when
        // no in-model nextn runtime was already attached (in-model takes precedence).
        // Honors RNB_DRAFTER_MODEL override and RNB_MTP_DISABLE_AUTO_DRAFTER opt-out.
        load_stage!("external_drafter_probe", {
            if !engine.mtp_runtime_ready() && std::env::var("RNB_MTP_DISABLE_AUTO_DRAFTER").is_err()
            {
                let drafter_path = std::env::var("RNB_DRAFTER_MODEL")
                    .ok()
                    .map(std::path::PathBuf::from)
                    .or_else(|| crate::auto_drafter::find_sibling_drafter(path));

                if let Some(ref drafter_path) = drafter_path {
                    match rnb_mtp::drafter::load_drafter(drafter_path) {
                        Ok(drafter) => match engine.attach_external_drafter(drafter) {
                            Ok(()) => eprintln!(
                                "[INFO] external drafter attached from {}",
                                drafter_path.display()
                            ),
                            Err(e) => eprintln!("[WARN] external drafter attach failed: {e}"),
                        },
                        Err(e) => eprintln!(
                            "[WARN] failed to load drafter from {}: {}",
                            drafter_path.display(),
                            e
                        ),
                    }
                } else if std::env::var("RNB_MTP").is_ok() {
                    eprintln!("[INFO] no sibling drafter found, MTP disabled");
                }
            }
        });

        load_profile.finish_and_emit(load_profile_total_start);
        Ok(engine)
    }

    pub fn mock(tokenizer: Tokenizer, metadata: ModelMetadata) -> Self {
        let kv_cache = KVCache::new(
            metadata.num_layers,
            metadata.max_seq_len,
            metadata.num_kv_heads,
            metadata.head_dim,
        );
        Self {
            tokenizer,
            kv_cache,
            metadata,
            architecture: ModelArchitecture::LLaMA,
            host_memory_plan: rnb_runtime::policy::HostMemoryPlan::default(),
            weights: None,
            scratch: None,
            mtp: None,
            mtp_runtime: None,
            backend_runtime: EngineBackendRuntime::new(),
            packed_model: None,
            shadow_model: None,
            memtrace_step: std::sync::atomic::AtomicUsize::new(0),
            moe_section_decode_bytes: None,
            #[cfg(feature = "vulkan")]
            fullpath_token_embd_bound: false,
            last_layer_hidden_cached: Vec::new(),
        }
    }
}

fn register_moe_layers_with_backend(
    weights: &super::layer_weights::ModelWeights,
    _metadata: &ModelMetadata,
) -> crate::error::Result<()> {
    #[cfg(feature = "cuda")]
    {
        let mut qwen_registered = 0usize;
        let mut qwen_skipped = 0usize;
        let mut nemotron_registered = 0usize;
        let mut nemotron_skipped = 0usize;
        let mut register_qwen =
            |moe_w: &super::layer_weights::SharedExpertMoELayerWeights| -> crate::error::Result<()> {
                let Some(gate_all) = moe_w.gate_exps_bytes() else {
                    qwen_skipped += 1;
                    return Ok(());
                };
                let Some(up_all) = moe_w.up_exps_bytes() else {
                    qwen_skipped += 1;
                    return Ok(());
                };
                let Some(down_all) = moe_w.down_exps_bytes() else {
                    qwen_skipped += 1;
                    return Ok(());
                };
                if moe_w.gate_quant != rnb_loader::GGMLType::Q4_K
                    || moe_w.up_quant != rnb_loader::GGMLType::Q4_K
                    || !matches!(
                        moe_w.down_quant,
                        rnb_loader::GGMLType::Q4_K
                            | rnb_loader::GGMLType::Q5_K
                            | rnb_loader::GGMLType::Q6_K
                    )
                {
                    qwen_skipped += 1;
                    return Ok(());
                }
                if super::backend_runtime::qwen_moe_register_layer(
                    gate_all,
                    up_all,
                    down_all,
                    moe_w.down_quant,
                    moe_w.n_ff,
                    moe_w.n_embd,
                )? {
                    qwen_registered += 1;
                } else {
                    qwen_skipped += 1;
                }
                Ok(())
            };
        let qwen_moe_layers = weights
            .layers
            .iter()
            .filter_map(|layer| match layer {
                LayerType::Attention(w) => w.shared_expert_moe.as_ref(),
                LayerType::GatedDeltaNet(w) => w.shared_expert_moe.as_ref(),
                LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => None,
            })
            .collect::<Vec<_>>();
        let qwen_layer_cache = super::cuda_runtime::moe_layer_cache_enabled();
        let mut qwen_iter = qwen_moe_layers.iter().copied().collect::<Vec<_>>();
        let keep_qwen_layer_cache_after_prefill =
            std::env::var("RNB_CUDA_QWEN35_KEEP_MOE_LAYER_CACHE_AFTER_PREFILL")
                .ok()
                .as_deref()
                == Some("1");
        if qwen_layer_cache && keep_qwen_layer_cache_after_prefill {
            qwen_iter.reverse();
        }
        for moe_w in qwen_iter {
            register_qwen(moe_w)?;
        }
        let nemotron_order = nemotron_q5_registration_order(&weights.layers);
        let log_nemotron_layers = std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1");
        for layer_idx in nemotron_order {
            let layer = &weights.layers[layer_idx];
            let LayerType::NemotronMoE(moe_w) = layer else {
                continue;
            };
            if moe_w.expert_up.ggml_type != rnb_loader::GGMLType::Q5_0
                || !matches!(
                    moe_w.expert_down.ggml_type,
                    rnb_loader::GGMLType::Q5_1 | rnb_loader::GGMLType::Q8_0
                )
            {
                nemotron_skipped += 1;
                if log_nemotron_layers {
                    eprintln!(
                        "[cuda] Nemotron Q5 resident layer skipped L{} quant up={:?} down={:?}",
                        layer_idx, moe_w.expert_up.ggml_type, moe_w.expert_down.ggml_type
                    );
                }
                continue;
            }
            let Some(up_all) = moe_w.expert_up.data.as_bytes() else {
                nemotron_skipped += 1;
                continue;
            };
            let Some(down_all) = moe_w.expert_down.data.as_bytes() else {
                nemotron_skipped += 1;
                continue;
            };
            let n_expert = moe_w.router.rows.max(_metadata.expert_used_count).max(1);
            let n_ff = moe_w.expert_up.rows / n_expert;
            let registered = if moe_w.expert_down.ggml_type == rnb_loader::GGMLType::Q8_0
                && std::env::var("RNB_CUDA_NEMOTRON_Q5_Q8_LAYER_CACHE")
                    .ok()
                    .as_deref()
                    == Some("1")
            {
                super::backend_runtime::nemotron_q5_q8_register_layer(
                    up_all,
                    down_all,
                    n_expert,
                    n_ff,
                    _metadata.hidden_dim,
                )?
            } else if moe_w.expert_down.ggml_type == rnb_loader::GGMLType::Q5_1 {
                super::backend_runtime::nemotron_q5_register_layer(
                    up_all,
                    down_all,
                    n_expert,
                    n_ff,
                    _metadata.hidden_dim,
                )?
            } else {
                false
            };
            if registered {
                nemotron_registered += 1;
                if log_nemotron_layers {
                    eprintln!(
                        "[cuda] Nemotron Q5 resident layer registered L{} bytes={}MiB",
                        layer_idx,
                        (up_all.len() + down_all.len()) / (1024 * 1024)
                    );
                }
            } else {
                nemotron_skipped += 1;
                if log_nemotron_layers {
                    eprintln!(
                        "[cuda] Nemotron Q5 resident layer skipped L{} bytes={}MiB",
                        layer_idx,
                        (up_all.len() + down_all.len()) / (1024 * 1024)
                    );
                }
            }
        }
        if qwen_registered > 0 {
            eprintln!(
                "[INFO] CUDA MoE layer residency: registered={} offloaded={}",
                qwen_registered, qwen_skipped
            );
        }
        if nemotron_registered > 0 {
            eprintln!(
                "[INFO] CUDA Nemotron Q5 layer residency: registered={} offloaded={}",
                nemotron_registered, nemotron_skipped
            );
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = weights;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn nemotron_q5_registration_order(layers: &[LayerType]) -> Vec<usize> {
    let indices = layers
        .iter()
        .enumerate()
        .filter_map(|(idx, layer)| matches!(layer, LayerType::NemotronMoE(_)).then_some(idx))
        .collect::<Vec<_>>();
    if let Ok(raw) = std::env::var("RNB_CUDA_NEMOTRON_Q5_LAYER_ORDER_LIST") {
        let mut ordered = Vec::with_capacity(indices.len());
        for item in raw.split(',') {
            let Ok(layer_idx) = item.trim().parse::<usize>() else {
                continue;
            };
            if indices.contains(&layer_idx) && !ordered.contains(&layer_idx) {
                ordered.push(layer_idx);
            }
        }
        for layer_idx in spread_order(indices) {
            if !ordered.contains(&layer_idx) {
                ordered.push(layer_idx);
            }
        }
        return ordered;
    }
    match std::env::var("RNB_CUDA_NEMOTRON_Q5_LAYER_ORDER")
        .ok()
        .as_deref()
    {
        Some("prefix") => indices,
        Some("tail") => indices.into_iter().rev().collect(),
        Some("hot") => hot_nemotron_order(indices),
        _ => spread_order(indices),
    }
}

#[cfg(feature = "cuda")]
fn spread_order(indices: Vec<usize>) -> Vec<usize> {
    if indices.len() <= 2 {
        return indices;
    }
    let mut ordered = Vec::with_capacity(indices.len());
    let mut left = 0usize;
    let mut right = indices.len() - 1;
    while left <= right {
        ordered.push(indices[left]);
        if left != right {
            ordered.push(indices[right]);
        }
        left += 1;
        right = right.saturating_sub(1);
    }
    ordered
}

#[cfg(feature = "cuda")]
fn hot_nemotron_order(indices: Vec<usize>) -> Vec<usize> {
    const HOT_LAYERS: &[usize] = &[43, 51, 34, 40, 22, 29, 17, 31, 38, 27, 36, 20];
    let mut ordered = Vec::with_capacity(indices.len());
    for &layer_idx in HOT_LAYERS {
        if indices.contains(&layer_idx) {
            ordered.push(layer_idx);
        }
    }
    for layer_idx in spread_order(indices) {
        if !ordered.contains(&layer_idx) {
            ordered.push(layer_idx);
        }
    }
    ordered
}
