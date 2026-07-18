use std::process::ExitCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmokeSuite {
    All,
    Gdn,
    Attention,
    Roundtrip,
}

fn parse_requested_suites(args: &[&str]) -> Result<Vec<SmokeSuite>, String> {
    if args.is_empty() {
        return Ok(vec![SmokeSuite::All]);
    }
    args.iter()
        .map(|arg| match *arg {
            "all" => Ok(SmokeSuite::All),
            "gdn" => Ok(SmokeSuite::Gdn),
            "attention" => Ok(SmokeSuite::Attention),
            "roundtrip" => Ok(SmokeSuite::Roundtrip),
            other => Err(format!(
                "unknown suite '{other}' (expected all, gdn, attention, or roundtrip)"
            )),
        })
        .collect()
}

fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let arg_refs = args.iter().map(String::as_str).collect::<Vec<_>>();
    let suites = match parse_requested_suites(&arg_refs) {
        Ok(suites) => suites,
        Err(e) => {
            eprintln!("{e}");
            eprintln!("usage: rnb-vulkan-smoke [all|gdn|attention|roundtrip]...");
            return ExitCode::from(2);
        }
    };

    match run_smoke_suites(&suites) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[vulkan-smoke] failed: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(feature = "vulkan")]
fn run_smoke_suites(suites: &[SmokeSuite]) -> Result<(), String> {
    let mut vk = rnb_backend_vulkan::VulkanLayerGemv::new(
        1024,
        248_320,
        64,
        rnb_backend_vulkan::GpuWeightMode::Soa,
    )?;
    let expanded = if suites.contains(&SmokeSuite::All) {
        vec![
            SmokeSuite::Gdn,
            SmokeSuite::Attention,
            SmokeSuite::Roundtrip,
        ]
    } else {
        suites.to_vec()
    };

    for suite in expanded {
        match suite {
            SmokeSuite::All => {}
            SmokeSuite::Gdn => run_gdn_smoke(&mut vk)?,
            SmokeSuite::Attention => run_attention_smoke(&mut vk)?,
            SmokeSuite::Roundtrip => run_roundtrip_smoke(&mut vk)?,
        }
    }
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn run_smoke_suites(_suites: &[SmokeSuite]) -> Result<(), String> {
    Err("rnb-vulkan-smoke requires --features vulkan".into())
}

#[cfg(feature = "vulkan")]
fn run_gdn_smoke(vk: &mut rnb_backend_vulkan::VulkanLayerGemv) -> Result<(), String> {
    eprintln!("[vulkan-smoke] gdn_delta_step");
    vk.self_test_gdn_delta_step()?;
    eprintln!("[vulkan-smoke] gdn_gated_norm_silu");
    vk.self_test_gdn_gated_norm_silu()?;
    eprintln!("[vulkan-smoke] gdn_qkv_conv_window");
    vk.self_test_gdn_qkv_conv_window()?;
    eprintln!("[vulkan-smoke] gdn_qkv_conv_window_resident_conv_state");
    vk.self_test_gdn_qkv_conv_window_resident_conv_state()?;
    eprintln!("[vulkan-smoke] gdn_qkv_conv_window_resident_conv_state_strided");
    vk.self_test_gdn_qkv_conv_window_resident_conv_state_strided()?;
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_attention_smoke(vk: &mut rnb_backend_vulkan::VulkanLayerGemv) -> Result<(), String> {
    eprintln!("[vulkan-smoke] q_window_into_kv_mirror_and_decode_grouped");
    vk.self_test_q_window_into_kv_mirror_and_decode_grouped()?;
    eprintln!("[vulkan-smoke] q_window_into_kv_mirror_and_decode_grouped_combined");
    vk.self_test_q_window_into_kv_mirror_and_decode_grouped_combined()?;
    eprintln!("[vulkan-smoke] gated_q_norm_rope_chain");
    vk.self_test_gated_q_norm_rope_chain()?;
    eprintln!("[vulkan-smoke] attention_decode_window");
    vk.self_test_attention_decode_window()?;
    Ok(())
}

#[cfg(feature = "vulkan")]
fn run_roundtrip_smoke(vk: &mut rnb_backend_vulkan::VulkanLayerGemv) -> Result<(), String> {
    eprintln!("[vulkan-smoke] q8_0_gemv_nonzero");
    vk.self_test_q8_0_gemv_nonzero()?;
    eprintln!("[vulkan-smoke] q8_0_gemv_multiblock_matches_cpu");
    vk.self_test_q8_0_gemv_multiblock_matches_cpu()?;
    eprintln!("[vulkan-smoke] q4k_block_parallel");
    let q4k_diff = vk.self_test_q4k_block_parallel()?;
    if q4k_diff >= 0.05 {
        return Err(format!("q4k_block_parallel max_diff too high: {q4k_diff}"));
    }
    eprintln!("[vulkan-smoke] argmax_pairs_f32_large_count");
    vk.self_test_argmax_pairs_f32_large_count()?;
    eprintln!("[vulkan-smoke] prefill_hidden_roundtrip");
    vk.self_test_prefill_hidden_roundtrip()?;
    eprintln!("[vulkan-smoke] prefill_hidden_offset_writes");
    vk.self_test_prefill_hidden_offset_writes()?;
    eprintln!("[vulkan-smoke] q_window_into_kv_mirror_avoids_kv_host_roundtrip");
    vk.self_test_q_window_into_kv_mirror_avoids_kv_host_roundtrip()?;
    eprintln!("[vulkan-smoke] q_window_decode_project_avoids_attn_host_roundtrip");
    vk.self_test_q_window_decode_project_avoids_attn_host_roundtrip()?;
    eprintln!("[vulkan-smoke] q_window_decode_project_elides_q_host_download");
    vk.self_test_q_window_decode_project_elides_q_host_download()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_as_all_suite() {
        let suites = parse_requested_suites(&[]).unwrap();

        assert_eq!(suites, vec![SmokeSuite::All]);
    }

    #[test]
    fn parses_named_suites_in_order() {
        let suites = parse_requested_suites(&["gdn", "attention"]).unwrap();

        assert_eq!(suites, vec![SmokeSuite::Gdn, SmokeSuite::Attention]);
    }

    #[test]
    fn rejects_unknown_suite() {
        let err = parse_requested_suites(&["nope"]).unwrap_err();

        assert!(err.contains("unknown suite"));
    }
}
