use std::path::Path;

use super::backend_runtime::{
    init_engine_backend_runtime, reset_backend_state_for_engine_init, EngineBackendRuntime,
};
use super::layer_weights::LayerType;
use super::layout::resolve_attention_layout;
use super::load_profile::LoadProfile;
use super::model_init::{build_model_metadata, build_tokenizer};
use super::models::shared_expert_moe::wire_sparse_expert_page_cache;
use super::mtp::{EngineMtpRuntime, EngineMtpState, InModelMtpRuntime};
use super::state::Engine;
use super::threading::configure_cpu_runtime;
use super::types::{ModelMetadata, ScratchBuffers};
use super::weight_loading::{load_model_weights, load_mtp_layer_weights};
use crate::kv_cache::{KVCache, KvCacheFormat};
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
fn default_kv_cache_format(
    has_kvarn_accelerator: bool,
    has_unaccelerated_backend: bool,
) -> KvCacheFormat {
    if has_unaccelerated_backend && !has_kvarn_accelerator {
        KvCacheFormat::F16
    } else {
        KvCacheFormat::KvarnK4V4G128
    }
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
    pub host_memory_budget: Option<crate::engine::memory_runtime::MemoryBudget>,
    pub kv_cache_format: Option<KvCacheFormat>,
}

impl EngineLoadConfig {
    pub fn with_host_ram_budget_bytes(mut self, bytes: u64) -> Self {
        self.host_memory_budget = (bytes > 0).then(|| {
            crate::engine::memory_runtime::MemoryBudget::new(
                crate::engine::memory_runtime::MemoryTier::Ram,
                bytes,
                0,
            )
        });
        self
    }

    pub fn with_kv_cache_format(mut self, format: KvCacheFormat) -> Self {
        self.kv_cache_format = Some(format);
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
            host_memory_budget,
            kv_cache_format,
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
            let architecture_for_thread_policy = rnb_loader::detect_model_architecture(path).ok();
            configure_cpu_runtime(architecture_for_thread_policy)
        })?;

        let model = load_stage!("load_model", { rnb_loader::load_model(path) })
            .map_err(|e| crate::error::LlmError::ModelLoad(e.to_string()))?;

        // Sum mapped tensor payloads rather than the selected path's file
        // length. A split GGUF's first shard may be a tiny metadata/tensor
        // index file while the remaining shards hold hundreds of GiB.
        let gguf_mapped_weight_bytes = model.weights.values().fold(0u64, |total, tensor| {
            total.saturating_add(tensor.as_bytes().map_or(0, |bytes| bytes.len() as u64))
        });
        let mapped_weight_bytes = gguf_mapped_weight_bytes;
        let cuda_memory_bytes = detected_cuda_memory_bytes();
        let host_memory_plan = crate::engine::policy::HostMemoryPlan::automatic(
            host_memory_budget,
            mapped_weight_bytes,
            cuda_memory_bytes.is_some(),
        );
        let sparse_moe_cuda_enabled = crate::engine::policy::cuda_q2k_q3k_sparse_moe_enabled(
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

        // Pre-dequantize all weights
        let mut weights = load_stage!("load_model_weights", {
            load_model_weights(
                &model,
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
                        LayerType::Attention(_) => metadata
                            .head_count_kv_per_layer
                            .as_ref()
                            .and_then(|v| v.get(layer_idx).copied())
                            .unwrap_or(metadata.num_kv_heads),
                        LayerType::GatedDeltaNet(_)
                        | LayerType::NemotronMamba2(_)
                        | LayerType::NemotronMoE(_) => 0,
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

        let has_kvarn_accelerator = cfg!(any(feature = "cuda", feature = "metal"));
        let has_unaccelerated_backend = cfg!(any(
            feature = "vulkan",
            feature = "opencl",
            feature = "mediatek"
        ));
        let kv_cache_format = match kv_cache_format {
            Some(format) => format,
            None => crate::engine::policy::env_string("RNB_KV_CACHE_FORMAT")
                .map(|value| value.parse::<KvCacheFormat>())
                .transpose()
                .map_err(crate::error::LlmError::ModelLoad)?
                .unwrap_or_else(|| {
                    // pm124: 압축 KVarn decode 는 매 토큰 전체 context re-dequant 가
                    // compute floor (측정 Metal: KVarn ~42 vs F16 67 t/s, +55%). 메모리
                    // 여유가 충분하면 F16(fast, memory-bound coalesced read)을, 부족하면
                    // KVarn(압축, offloading·장문)을 고르는 적응형 정책. F16 KV 가 가중치
                    // 로드 후 남는 예산의 절반 이하(2x headroom)일 때만 F16 — 낮은 RAM /
                    // 장문에서는 자동으로 KVarn 로 후퇴. Metal 한정(CUDA 기본은 불변).
                    let f16_kv_bytes: u64 = weights
                        .layers
                        .iter()
                        .enumerate()
                        .filter(|(idx, layer)| {
                            matches!(layer, LayerType::Attention(_)) && layer_num_kv_heads[*idx] > 0
                        })
                        .map(|(idx, _)| {
                            (metadata.max_seq_len as u64)
                                * (layer_num_kv_heads[idx] as u64)
                                * (layer_head_dims[idx] as u64)
                                * 2 // k + v
                                * 2 // f16 bytes
                        })
                        .sum();
                    let f16_fits = cfg!(all(feature = "metal", not(feature = "cuda")))
                        && f16_kv_bytes > 0
                        && host_memory_plan.ram_budget().is_some_and(|budget| {
                            let remaining = budget
                                .available_bytes()
                                .saturating_sub(mapped_weight_bytes as u64);
                            f16_kv_bytes.saturating_mul(2) <= remaining
                        });
                    if f16_fits {
                        KvCacheFormat::F16
                    } else {
                        default_kv_cache_format(has_kvarn_accelerator, has_unaccelerated_backend)
                    }
                }),
        };
        let layer_kv_formats = weights
            .layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| match layer {
                LayerType::Attention(_)
                    if layer_num_kv_heads[layer_idx] > 0
                        && layer_head_dims[layer_idx] >= 4
                        && layer_head_dims[layer_idx] % 4 == 0
                        && weights
                            .glm_dsa_attention
                            .as_ref()
                            .and_then(|layers| layers.get(layer_idx))
                            .is_none() =>
                {
                    kv_cache_format
                }
                _ => KvCacheFormat::F16,
            })
            .collect::<Vec<_>>();

        let mut kv_cache = load_stage!("kv_cache_alloc", {
            KVCache::new_per_layer_with_formats(
                metadata.max_seq_len,
                &layer_num_kv_heads,
                &layer_head_dims,
                &layer_kv_formats,
            )
            .map_err(crate::error::LlmError::ModelLoad)?
        });
        if kv_cache_format != KvCacheFormat::F16 {
            eprintln!(
                "[kv-cache] format={} attention_layers={} capacity={:.2} MiB allocated={:.2} MiB",
                kv_cache_format.label(),
                layer_kv_formats
                    .iter()
                    .filter(|&&format| format == kv_cache_format)
                    .count(),
                kv_cache.capacity_kv_bytes() as f64 / (1024.0 * 1024.0),
                kv_cache.allocated_kv_bytes() as f64 / (1024.0 * 1024.0),
            );
        }

        // pm119 2단계: GLM DSA lightning indexer key 캐시 (opt-in
        // `RNB_GLM_DSA_INDEXER=1` — selected-set attention 통합 전 개발 게이트).
        #[cfg(feature = "cuda")]
        if crate::engine::policy::env_string("RNB_GLM_DSA_INDEXER").as_deref() == Some("1") {
            return Err(crate::error::LlmError::ModelLoad(
                "RNB_GLM_DSA_INDEXER is unavailable in CUDA builds because selected-set attention has no CUDA implementation and CPU fallback is disabled"
                    .into(),
            ));
        }
        #[cfg(not(feature = "cuda"))]
        if crate::engine::policy::env_string("RNB_GLM_DSA_INDEXER").as_deref() == Some("1") {
            if let (Some(layers), Some(indexer_meta)) =
                (&weights.glm_dsa_attention, model.metadata.glm_indexer)
            {
                if let Some(key_len) = layers.iter().find_map(|layer| layer.indexer_key_len()) {
                    kv_cache.init_glm_indexer(layers.len(), key_len, indexer_meta.top_k);
                    eprintln!(
                        "[INFO] GLM DSA indexer cache enabled: layers={} key_len={key_len} top_k={}",
                        layers.len(),
                        indexer_meta.top_k
                    );
                }
            }
        }

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

        let all_layers_attention = !weights.layers.is_empty()
            && weights
                .layers
                .iter()
                .all(|l| matches!(l, LayerType::Attention(_)));
        maybe_enable_q4k_prefill_f16_gemm_for_dense_attention(all_layers_attention);

        let backend_runtime = load_stage!("backend_runtime_init", {
            init_engine_backend_runtime(&metadata, &weights, ffn_inner_dim)
        });
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
                memtrace_step: std::sync::atomic::AtomicUsize::new(0),
                #[cfg(feature = "vulkan")]
                fullpath_token_embd_bound: false,
                last_layer_hidden_cached: Vec::new(),
                mtp_auto_requested_cache: std::sync::OnceLock::new(),
            }
        });
        load_stage!("axis_p_mlock", {
            engine.apply_axis_p_mlock();
        });
        load_stage!("output_weight_mlock", {
            engine.apply_output_weight_mlock();
        });
        #[cfg(feature = "vulkan")]
        load_stage!("vulkan_fullpath_model_prepare", {
            engine.prepare_vulkan_fullpath_model_for_load()
        })?;

        // mc78 Phase 1: Auto-detect sibling external drafter. Only fires when
        // no in-model nextn runtime was already attached (in-model takes precedence).
        // Honors RNB_DRAFTER_MODEL override and RNB_MTP_DISABLE_AUTO_DRAFTER opt-out.
        load_stage!("external_drafter_probe", {
            if !engine.mtp_runtime_ready()
                && crate::engine::policy::env_string("RNB_MTP_DISABLE_AUTO_DRAFTER").is_none()
            {
                let drafter_path = crate::engine::policy::env_string("RNB_DRAFTER_MODEL")
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
                } else if crate::engine::policy::env_string("RNB_MTP").is_some() {
                    eprintln!("[INFO] no sibling drafter found, MTP disabled");
                }
            }
        });

        #[cfg(feature = "cuda")]
        load_stage!("mtp_device_verify_prepare", {
            if engine.mtp_spec_requested() && engine.mtp_device_verify_requested() {
                engine.prewarm_mtp_device_verify_static_weights()?;
                if let Some(scratch) = engine.scratch.as_mut() {
                    scratch.device_verify_static_weights_warmed = true;
                }
            }
        });

        let qwen_moe_runtime_headroom_bytes = {
            #[cfg(feature = "cuda")]
            {
                if engine.mtp_spec_requested() && engine.mtp_device_verify_requested() {
                    engine
                        .mtp_auto_policy()
                        .min_free_vram_mib
                        .saturating_mul(1024 * 1024)
                } else {
                    0
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                0
            }
        };

        load_stage!("backend_register_moe", {
            let weights = engine
                .weights
                .as_ref()
                .expect("loaded engine must retain model weights");
            register_moe_layers_with_backend(
                weights,
                &engine.metadata,
                engine.mtp_runtime.as_ref(),
                qwen_moe_runtime_headroom_bytes,
            )
        })?;

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
            host_memory_plan: crate::engine::policy::HostMemoryPlan::default(),
            weights: None,
            scratch: None,
            mtp: None,
            mtp_runtime: None,
            backend_runtime: EngineBackendRuntime::new(),
            memtrace_step: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "vulkan")]
            fullpath_token_embd_bound: false,
            last_layer_hidden_cached: Vec::new(),
            mtp_auto_requested_cache: std::sync::OnceLock::new(),
        }
    }
}

fn register_moe_layers_with_backend(
    weights: &super::layer_weights::ModelWeights,
    _metadata: &ModelMetadata,
    _mtp_runtime: Option<&super::mtp::EngineMtpRuntime>,
    _runtime_headroom_bytes: usize,
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
        let mut qwen_moe_layers = weights
            .layers
            .iter()
            .filter_map(|layer| match layer {
                LayerType::Attention(w) => w.shared_expert_moe.as_ref(),
                LayerType::GatedDeltaNet(w) => w.shared_expert_moe.as_ref(),
                LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => None,
            })
            .collect::<Vec<_>>();
        if let Some(moe) = _mtp_runtime.and_then(super::mtp::EngineMtpRuntime::shared_expert_moe) {
            qwen_moe_layers.push(moe);
        }
        let (qwen_moe_cacheable_layers, qwen_moe_bytes, qwen_moe_min_layer_bytes) = qwen_moe_layers
            .iter()
            .filter(|moe_w| {
                moe_w.gate_quant == rnb_loader::GGMLType::Q4_K
                    && moe_w.up_quant == rnb_loader::GGMLType::Q4_K
                    && matches!(
                        moe_w.down_quant,
                        rnb_loader::GGMLType::Q4_K
                            | rnb_loader::GGMLType::Q5_K
                            | rnb_loader::GGMLType::Q6_K
                    )
            })
            .filter_map(|moe_w| {
                Some(
                    moe_w
                        .gate_exps_bytes()?
                        .len()
                        .saturating_add(moe_w.up_exps_bytes()?.len())
                        .saturating_add(moe_w.down_exps_bytes()?.len()),
                )
            })
            .fold(
                (0usize, 0usize, usize::MAX),
                |(layers, bytes, min_layer_bytes), layer_bytes| {
                    (
                        layers.saturating_add(1),
                        bytes.saturating_add(layer_bytes),
                        min_layer_bytes.min(layer_bytes),
                    )
                },
            );
        let qwen_moe_min_layer_bytes = (qwen_moe_cacheable_layers > 0)
            .then_some(qwen_moe_min_layer_bytes)
            .unwrap_or(0);
        let env_bool_or = |name: &str, default: bool| {
            crate::engine::policy::env_string(name)
                .map(|value| {
                    !matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "0" | "false" | "off" | "no"
                    )
                })
                .unwrap_or(default)
        };
        let qwen_cache_explicit = qwen_moe_cache_explicit_override(
            crate::engine::policy::env_string("RNB_CUDA_MOE_LAYER_CACHE").as_deref(),
            crate::engine::policy::env_string("RNB_CUDA_MOE_LAYER_CACHE_MB").as_deref(),
        );
        let mut qwen_layer_cache = env_bool_or("RNB_CUDA_MOE_LAYER_CACHE", qwen_moe_bytes > 0);
        if qwen_layer_cache && qwen_moe_bytes > 0 {
            let cache_limit = super::backend_runtime::qwen_moe_configure_layer_cache(
                qwen_moe_bytes,
                _runtime_headroom_bytes,
                qwen_moe_min_layer_bytes,
            )?;
            qwen_layer_cache =
                qwen_moe_layer_cache_enabled_for_budget(true, qwen_moe_bytes, cache_limit);
            if !qwen_layer_cache
                && crate::engine::policy::env_string("RNB_CUDA_CACHE_LOG").as_deref() == Some("1")
            {
                eprintln!(
                    "[cuda] Qwen MoE full-layer cache fallback: model={}MiB limit={}MiB",
                    qwen_moe_bytes / (1024 * 1024),
                    cache_limit / (1024 * 1024),
                );
            }
        }
        let mut qwen_iter = qwen_moe_layers.iter().copied().collect::<Vec<_>>();
        let keep_qwen_layer_cache_after_prefill = env_bool_or(
            "RNB_CUDA_QWEN35_KEEP_MOE_LAYER_CACHE_AFTER_PREFILL",
            qwen_layer_cache,
        );
        if qwen_layer_cache && keep_qwen_layer_cache_after_prefill {
            qwen_iter.reverse();
        }
        if qwen_layer_cache {
            let mut automatic_fallback_reason = None;
            for moe_w in qwen_iter {
                if let Err(err) = register_qwen(moe_w) {
                    if qwen_cache_explicit {
                        return Err(err);
                    }
                    automatic_fallback_reason = Some(format!("registration failed: {err}"));
                    break;
                }
            }
            if !qwen_cache_explicit
                && automatic_fallback_reason.is_none()
                && qwen_registered == 0
                && qwen_moe_cacheable_layers > 0
            {
                automatic_fallback_reason = Some(format!(
                    "registered no cacheable layer out of {qwen_moe_cacheable_layers}"
                ));
            }
            if let Some(reason) = automatic_fallback_reason {
                crate::engine::cuda_runtime::clear_moe_layer_cache()
                    .map_err(crate::error::LlmError::Forward)?;
                qwen_registered = 0;
                qwen_skipped = qwen_moe_layers.len();
                eprintln!("[WARN] CUDA Qwen MoE full-layer cache disabled: {reason}");
            }
        }
        let nemotron_order = nemotron_q5_registration_order(&weights.layers);
        let log_nemotron_layers =
            crate::engine::policy::env_string("RNB_CUDA_CACHE_LOG").as_deref() == Some("1");
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
                && crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_Q5_Q8_LAYER_CACHE")
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
fn qwen_moe_cache_explicit_override(cache_flag: Option<&str>, cache_mb: Option<&str>) -> bool {
    cache_flag.is_some_and(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    }) || cache_mb
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .is_some()
}

#[cfg(feature = "cuda")]
fn qwen_moe_layer_cache_enabled_for_budget(
    configured_enabled: bool,
    model_moe_bytes: usize,
    cache_limit: usize,
) -> bool {
    configured_enabled && model_moe_bytes > 0 && cache_limit > 0
}

#[cfg(feature = "cuda")]
fn nemotron_q5_registration_order(layers: &[LayerType]) -> Vec<usize> {
    let indices = layers
        .iter()
        .enumerate()
        .filter_map(|(idx, layer)| matches!(layer, LayerType::NemotronMoE(_)).then_some(idx))
        .collect::<Vec<_>>();
    if let Some(raw) = crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_Q5_LAYER_ORDER_LIST") {
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
    match crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_Q5_LAYER_ORDER").as_deref() {
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

#[cfg(test)]
mod kv_cache_default_tests {
    use super::*;

    #[test]
    fn native_cpu_defaults_to_k4v4_g128() {
        assert_eq!(
            default_kv_cache_format(false, false),
            KvCacheFormat::KvarnK4V4G128
        );
    }

    #[test]
    fn cuda_and_metal_builds_default_to_k4v4_g128() {
        assert_eq!(
            default_kv_cache_format(true, false),
            KvCacheFormat::KvarnK4V4G128
        );
        assert_eq!(
            default_kv_cache_format(true, true),
            KvCacheFormat::KvarnK4V4G128
        );
    }

    #[test]
    fn unaccelerated_gpu_backends_keep_f16_default() {
        assert_eq!(default_kv_cache_format(false, true), KvCacheFormat::F16);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn qwen_moe_cache_auto_uses_available_partial_budget() {
        assert!(qwen_moe_layer_cache_enabled_for_budget(
            true, 18_000, 18_000
        ));
        assert!(qwen_moe_layer_cache_enabled_for_budget(true, 18_000, 8_000));
        assert!(!qwen_moe_layer_cache_enabled_for_budget(true, 18_000, 0));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn qwen_moe_cache_auto_values_are_not_explicit_overrides() {
        assert!(!qwen_moe_cache_explicit_override(None, None));
        assert!(!qwen_moe_cache_explicit_override(
            Some("auto"),
            Some("auto")
        ));
        assert!(qwen_moe_cache_explicit_override(Some("1"), None));
        assert!(qwen_moe_cache_explicit_override(None, Some("8192")));
    }
}
