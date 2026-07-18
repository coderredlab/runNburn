//! Quantized decode dispatch helpers.

use super::*;

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn decode_ffn_up_cpu_best_effort<F>(
    scratch: &mut ScratchBuffers,
    up_weight: &QuantizedWeight,
    hidden_dim: usize,
    mut profile: F,
) where
    F: FnMut(&'static str, Instant),
{
    let input = &scratch.norm_buf[..hidden_dim];

    #[cfg(target_arch = "aarch64")]
    {
        let has_dotprod = fast_dotprod_enabled();
        let use_q8k = has_dotprod && k_quant_q8k_candidate(up_weight);
        if use_q8k {
            gemm_runtime::quantize_input_q8k_into(input, &mut scratch.arch_scratch.q8k_scratch);
            let t_up = Instant::now();
            up_weight
                .gemv_into_q8k(&scratch.arch_scratch.q8k_scratch, &mut scratch.ffn_up)
                .ok();
            profile("up_gemv", t_up);
        } else if has_dotprod {
            gemm_runtime::quantize_input_q8_into(input, &mut scratch.arch_scratch.q8_scratch);
            let t_up = Instant::now();
            if up_weight
                .gemv_into_q8(&scratch.arch_scratch.q8_scratch, &mut scratch.ffn_up)
                .is_err()
            {
                up_weight.gemv_into(input, &mut scratch.ffn_up).ok();
            }
            profile("up_gemv", t_up);
        } else {
            let t_up = Instant::now();
            up_weight.gemv_into(input, &mut scratch.ffn_up).ok();
            profile("up_gemv", t_up);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let t_up = Instant::now();
        up_weight.gemv_into(input, &mut scratch.ffn_up).ok();
        profile("up_gemv", t_up);
    }
}

pub(in crate::engine) fn decode_ffn_gate_up_cpu_into<F>(
    scratch: &mut ScratchBuffers,
    architecture: ModelArchitecture,
    gate_weight: &QuantizedWeight,
    up_weight: &QuantizedWeight,
    fused_gate_up: Option<&QuantizedWeight>,
    hidden_dim: usize,
    mut profile: F,
) -> crate::error::Result<()>
where
    F: FnMut(&'static str, Instant),
{
    let input = &scratch.norm_buf[..hidden_dim];

    #[cfg(target_arch = "aarch64")]
    {
        let has_dotprod = fast_dotprod_enabled();
        let use_q8k = has_dotprod && k_quant_q8k_candidate(gate_weight);

        if use_q8k {
            gemm_runtime::quantize_input_q8k_into(input, &mut scratch.arch_scratch.q8k_scratch);
            let t_gate = Instant::now();
            gate_weight.gemv_into_q8k(&scratch.arch_scratch.q8k_scratch, &mut scratch.ffn_gate)?;
            profile("gate_gemv", t_gate);
            let t_up = Instant::now();
            up_weight.gemv_into_q8k(&scratch.arch_scratch.q8k_scratch, &mut scratch.ffn_up)?;
            profile("up_gemv", t_up);
            let t_act = Instant::now();
            let gate_rows = gate_weight.rows;
            apply_model_gate_mul_inplace(
                &mut scratch.ffn_gate[..gate_rows],
                &scratch.ffn_up[..gate_rows],
                architecture,
            );
            profile("silu_mul", t_act);
            return Ok(());
        }

        if has_dotprod {
            gemm_runtime::quantize_input_q8_into(input, &mut scratch.arch_scratch.q8_scratch);
            if let Some(fused) = fused_gate_up {
                let gate_rows = gate_weight.rows;
                let t_fused = Instant::now();
                fused.gemv_into_q8(
                    &scratch.arch_scratch.q8_scratch,
                    &mut scratch.arch_scratch.ffn_combined,
                )?;
                profile("gate_up_fused_gemv", t_fused);
                scratch.ffn_gate[..gate_rows]
                    .copy_from_slice(&scratch.arch_scratch.ffn_combined[..gate_rows]);
                scratch.ffn_up[..gate_rows]
                    .copy_from_slice(&scratch.arch_scratch.ffn_combined[gate_rows..gate_rows * 2]);
                let t_act = Instant::now();
                apply_model_gate_mul_inplace(
                    &mut scratch.ffn_gate[..gate_rows],
                    &scratch.ffn_up[..gate_rows],
                    architecture,
                );
                profile("silu_mul", t_act);
                return Ok(());
            }

            let t_gate_up = Instant::now();
            if gate_weight
                .gemv_into_q8(&scratch.arch_scratch.q8_scratch, &mut scratch.ffn_gate)
                .is_ok()
                && up_weight
                    .gemv_into_q8(&scratch.arch_scratch.q8_scratch, &mut scratch.ffn_up)
                    .is_ok()
            {
                profile("gate+up_gemv", t_gate_up);
                let t_act = Instant::now();
                apply_model_gate_mul_inplace(&mut scratch.ffn_gate, &scratch.ffn_up, architecture);
                profile("silu_mul", t_act);
                return Ok(());
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    let _ = fused_gate_up;

    let t_gate = Instant::now();
    gate_weight.gemv_into(input, &mut scratch.ffn_gate)?;
    profile("gate_gemv", t_gate);
    let t_up = Instant::now();
    up_weight.gemv_into(input, &mut scratch.ffn_up)?;
    profile("up_gemv", t_up);
    let t_act = Instant::now();
    apply_model_gate_mul_inplace(&mut scratch.ffn_gate, &scratch.ffn_up, architecture);
    profile("silu_mul", t_act);
    Ok(())
}

pub(in crate::engine) fn decode_attention_qkv_cpu_into<F>(
    scratch: &mut ScratchBuffers,
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    hidden_dim: usize,
    q_out_dim: usize,
    kv_dim: usize,
    verbose: bool,
    mut profile: F,
) -> crate::error::Result<()>
where
    F: FnMut(&'static str, Instant),
{
    let norm = &scratch.norm_buf[..hidden_dim];

    #[cfg(target_arch = "aarch64")]
    let attn_q8k = if decode_q8k_candidate(q_weight, hidden_dim, false) {
        gemm_runtime::quantize_input_q8k_into(norm, &mut scratch.arch_scratch.q8k_scratch);
        Some(&scratch.arch_scratch.q8k_scratch[..])
    } else {
        None
    };

    if verbose {
        let t_q = Instant::now();
        #[cfg(target_arch = "aarch64")]
        if let Some(ref q8k) = attn_q8k {
            q_weight.gemv_into_q8k(q8k, &mut scratch.q_buf[..q_out_dim])?;
        } else {
            q_weight.gemv_into(norm, &mut scratch.q_buf[..q_out_dim])?;
        }
        #[cfg(not(target_arch = "aarch64"))]
        q_weight.gemv_into(norm, &mut scratch.q_buf[..q_out_dim])?;
        profile("q_weight", t_q);

        let t_k = Instant::now();
        #[cfg(target_arch = "aarch64")]
        if let Some(ref q8k) = attn_q8k {
            k_weight.gemv_into_q8k(q8k, &mut scratch.k_buf[..kv_dim])?;
        } else {
            k_weight.gemv_into(norm, &mut scratch.k_buf[..kv_dim])?;
        }
        #[cfg(not(target_arch = "aarch64"))]
        k_weight.gemv_into(norm, &mut scratch.k_buf[..kv_dim])?;
        profile("k_weight", t_k);

        let t_v = Instant::now();
        #[cfg(target_arch = "aarch64")]
        if let Some(ref q8k) = attn_q8k {
            v_weight.gemv_into_q8k(q8k, &mut scratch.v_buf[..kv_dim])?;
        } else {
            v_weight.gemv_into(norm, &mut scratch.v_buf[..kv_dim])?;
        }
        #[cfg(not(target_arch = "aarch64"))]
        v_weight.gemv_into(norm, &mut scratch.v_buf[..kv_dim])?;
        profile("v_weight", t_v);
        return Ok(());
    }

    let q_ptr = scratch.q_buf.as_mut_ptr() as usize;
    let k_ptr = scratch.k_buf.as_mut_ptr() as usize;
    let v_ptr = scratch.v_buf.as_mut_ptr() as usize;

    #[cfg(target_arch = "aarch64")]
    {
        let (r1, r2) = if let Some(ref q8k) = attn_q8k {
            rayon::join(
                || {
                    q_weight.gemv_into_q8k(q8k, unsafe {
                        std::slice::from_raw_parts_mut(q_ptr as *mut f32, q_out_dim)
                    })
                },
                || {
                    k_weight.gemv_into_q8k(q8k, unsafe {
                        std::slice::from_raw_parts_mut(k_ptr as *mut f32, kv_dim)
                    })?;
                    v_weight.gemv_into_q8k(q8k, unsafe {
                        std::slice::from_raw_parts_mut(v_ptr as *mut f32, kv_dim)
                    })
                },
            )
        } else {
            rayon::join(
                || {
                    q_weight.gemv_into(norm, unsafe {
                        std::slice::from_raw_parts_mut(q_ptr as *mut f32, q_out_dim)
                    })
                },
                || {
                    k_weight.gemv_into(norm, unsafe {
                        std::slice::from_raw_parts_mut(k_ptr as *mut f32, kv_dim)
                    })?;
                    v_weight.gemv_into(norm, unsafe {
                        std::slice::from_raw_parts_mut(v_ptr as *mut f32, kv_dim)
                    })
                },
            )
        };
        r1?;
        r2?;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (r1, r2) = rayon::join(
            || {
                q_weight.gemv_into(norm, unsafe {
                    std::slice::from_raw_parts_mut(q_ptr as *mut f32, q_out_dim)
                })
            },
            || {
                k_weight.gemv_into(norm, unsafe {
                    std::slice::from_raw_parts_mut(k_ptr as *mut f32, kv_dim)
                })?;
                v_weight.gemv_into(norm, unsafe {
                    std::slice::from_raw_parts_mut(v_ptr as *mut f32, kv_dim)
                })
            },
        );
        r1?;
        r2?;
    }
    Ok(())
}

pub(in crate::engine) fn decode_gdn_qkv_gate_cpu_into(
    scratch: &mut ScratchBuffers,
    qkv_weight: &QuantizedWeight,
    gate_weight: &QuantizedWeight,
    hidden_dim: usize,
    conv_channels: usize,
    d_inner: usize,
) -> crate::error::Result<(u64, u64)> {
    let norm = &scratch.norm_buf[..hidden_dim];
    let qkv_us = AtomicU64::new(0);
    let gate_us = AtomicU64::new(0);

    #[cfg(target_arch = "aarch64")]
    let norm_q8k = if decode_q8k_candidate(qkv_weight, hidden_dim, true) {
        gemm_runtime::quantize_input_q8k_into(norm, &mut scratch.arch_scratch.q8k_scratch);
        Some(&scratch.arch_scratch.q8k_scratch[..])
    } else {
        None
    };

    let qkv_ptr = scratch.qkv_buf.as_mut_ptr() as usize;
    let z_ptr = scratch.z_buf.as_mut_ptr() as usize;

    #[cfg(target_arch = "aarch64")]
    {
        let (r1, r2) = if let Some(ref q8k) = norm_q8k {
            rayon::join(
                || {
                    let t_qkv = Instant::now();
                    let res = qkv_weight.gemv_into_q8k(q8k, unsafe {
                        std::slice::from_raw_parts_mut(qkv_ptr as *mut f32, conv_channels)
                    });
                    qkv_us.store(t_qkv.elapsed().as_micros() as u64, Ordering::Relaxed);
                    res
                },
                || {
                    let t_gate = Instant::now();
                    let res = gate_weight.gemv_into_q8k(q8k, unsafe {
                        std::slice::from_raw_parts_mut(z_ptr as *mut f32, d_inner)
                    });
                    gate_us.store(t_gate.elapsed().as_micros() as u64, Ordering::Relaxed);
                    res
                },
            )
        } else {
            rayon::join(
                || {
                    let t_qkv = Instant::now();
                    let res = qkv_weight.gemv_into(norm, unsafe {
                        std::slice::from_raw_parts_mut(qkv_ptr as *mut f32, conv_channels)
                    });
                    qkv_us.store(t_qkv.elapsed().as_micros() as u64, Ordering::Relaxed);
                    res
                },
                || {
                    let t_gate = Instant::now();
                    let res = gate_weight.gemv_into(norm, unsafe {
                        std::slice::from_raw_parts_mut(z_ptr as *mut f32, d_inner)
                    });
                    gate_us.store(t_gate.elapsed().as_micros() as u64, Ordering::Relaxed);
                    res
                },
            )
        };
        r1?;
        r2?;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (r1, r2) = rayon::join(
            || {
                let t_qkv = Instant::now();
                let res = qkv_weight.gemv_into(norm, unsafe {
                    std::slice::from_raw_parts_mut(qkv_ptr as *mut f32, conv_channels)
                });
                qkv_us.store(t_qkv.elapsed().as_micros() as u64, Ordering::Relaxed);
                res
            },
            || {
                let t_gate = Instant::now();
                let res = gate_weight.gemv_into(norm, unsafe {
                    std::slice::from_raw_parts_mut(z_ptr as *mut f32, d_inner)
                });
                gate_us.store(t_gate.elapsed().as_micros() as u64, Ordering::Relaxed);
                res
            },
        );
        r1?;
        r2?;
    }

    Ok((
        qkv_us.load(Ordering::Relaxed),
        gate_us.load(Ordering::Relaxed),
    ))
}
