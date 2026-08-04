#[cfg(feature = "metal")]
use crate::engine::metal_runtime;

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_decode_legacy_carrier_enabled_by_policy() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_decode_parity_policy().legacy_carrier_enabled();
    }
    #[cfg(not(feature = "metal"))]
    true
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_decode_legacy_attn_layer_enabled_by_policy() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_decode_parity_policy().legacy_attn_layer_enabled;
    }
    #[cfg(not(feature = "metal"))]
    true
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_qwen_moe_decode_chain_enabled_by_policy() -> bool {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_decode_parity_policy().qwen_moe_decode_chain_enabled;
    }
    #[cfg(not(feature = "metal"))]
    false
}

#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_decode_kv_int8_requires_carrier_error(
    carrier_chain_enabled: bool,
    attn_layer_enabled: bool,
) -> Option<&'static str> {
    #[cfg(feature = "metal")]
    {
        return metal_runtime::metal_decode_parity_policy()
            .kv_int8_requires_carrier_error(carrier_chain_enabled, attn_layer_enabled);
    }
    #[cfg(not(feature = "metal"))]
    None
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_decode_parity_counters_reset() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_decode_parity_counters_reset();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(in crate::engine) fn metal_decode_parity_record_expected_token() {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_decode_parity_record_expected_token();
    }
}

#[cfg_attr(not(feature = "metal"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn metal_decode_parity_counters_report(label: &str) {
    #[cfg(feature = "metal")]
    {
        metal_runtime::metal_decode_parity_counters_report(label);
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = label;
    }
}
#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod metal_decode_policy_facade_tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_decode_env() {
        for key in [
            "RNB_METAL_DECODE_CHAIN",
            "RNB_METAL_GDN_LAYER",
            "RNB_METAL_ATTN_LAYER",
            "RNB_METAL_QWEN35_MOE_DECODE_CHAIN",
            "RNB_METAL_KV_INT8",
            "RNB_METAL_DECODE_PARITY_TIME",
        ] {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn metal_decode_policy_facade_reads_runtime_policy() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_decode_env();
        assert!(metal_decode_legacy_carrier_enabled_by_policy());
        assert!(metal_decode_legacy_attn_layer_enabled_by_policy());
        assert!(metal_qwen_moe_decode_chain_enabled_by_policy());
        assert_eq!(
            metal_decode_kv_int8_requires_carrier_error(true, true),
            None
        );

        std::env::set_var("RNB_METAL_DECODE_CHAIN", "0");
        std::env::set_var("RNB_METAL_GDN_LAYER", "0");
        std::env::set_var("RNB_METAL_ATTN_LAYER", "0");
        std::env::set_var("RNB_METAL_KV_INT8", "1");
        std::env::set_var("RNB_METAL_QWEN35_MOE_DECODE_CHAIN", "0");

        assert!(!metal_decode_legacy_carrier_enabled_by_policy());
        assert!(!metal_decode_legacy_attn_layer_enabled_by_policy());
        assert!(!metal_qwen_moe_decode_chain_enabled_by_policy());
        assert_eq!(
            metal_decode_kv_int8_requires_carrier_error(false, false),
            Some("RNB_METAL_KV_INT8=1 requires Metal carrier chain and attention layer")
        );
        std::env::set_var("RNB_METAL_QWEN35_MOE_DECODE_CHAIN", "1");
        assert!(metal_qwen_moe_decode_chain_enabled_by_policy());
        clear_decode_env();
    }

    #[test]
    fn metal_decode_parity_counter_facade_is_callable() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_decode_env();
        std::env::set_var("RNB_METAL_DECODE_PARITY_TIME", "1");

        metal_decode_parity_counters_reset();
        metal_decode_parity_record_expected_token();
        metal_decode_parity_counters_report("facade-test");

        clear_decode_env();
    }
}
