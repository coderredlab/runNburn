#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuantResidentBudgetPlan {
    pub enabled: bool,
    pub raw_quant_target_mib: usize,
    pub packed_promotion_target_mib: usize,
}

fn parse_quant_resident_env() -> Result<QuantResidentEnv, String> {
    let raw = match std::env::var("RNB_CUDA_QUANT_RESIDENT_MB") {
        Ok(raw) => Some(raw),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err("RNB_CUDA_QUANT_RESIDENT_MB must be valid UTF-8".to_string());
        }
    };
    parse_quant_resident_env_value(raw.as_deref())
}

fn parse_quant_resident_env_value(raw: Option<&str>) -> Result<QuantResidentEnv, String> {
    let Some(raw) = raw else {
        return Ok(QuantResidentEnv::Auto);
    };
    let raw = raw.trim();
    if raw.is_empty()
        || matches!(
            raw.to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        )
    {
        return Ok(QuantResidentEnv::Disabled);
    }
    if raw.eq_ignore_ascii_case("auto") {
        return Ok(QuantResidentEnv::Auto);
    }
    let mib = raw.parse::<usize>().map_err(|e| {
        format!("RNB_CUDA_QUANT_RESIDENT_MB must be auto, off, or integer MiB: {e}")
    })?;
    Ok(if mib == 0 {
        QuantResidentEnv::Disabled
    } else {
        QuantResidentEnv::FixedMib(mib)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantResidentEnv {
    Disabled,
    Auto,
    FixedMib(usize),
}

pub(crate) fn quant_resident_policy_requested() -> Result<bool, String> {
    parse_quant_resident_env().map(|env| !matches!(env, QuantResidentEnv::Disabled))
}

pub(crate) fn quant_resident_budget_plan(
    total_mib: usize,
    free_mib: usize,
    model_quant_mib: usize,
    packed_hot_extra_mib: usize,
) -> Result<QuantResidentBudgetPlan, String> {
    let env = parse_quant_resident_env()?;
    if matches!(env, QuantResidentEnv::Disabled) {
        return Ok(QuantResidentBudgetPlan {
            enabled: false,
            raw_quant_target_mib: 0,
            packed_promotion_target_mib: 0,
        });
    }

    let reserve_mib = quant_resident_reserve_mib(total_mib);
    let available_mib = free_mib.saturating_sub(reserve_mib);
    let requested_mib = match env {
        QuantResidentEnv::Disabled => 0,
        QuantResidentEnv::Auto => model_quant_mib.saturating_add(packed_hot_extra_mib),
        QuantResidentEnv::FixedMib(mib) => mib,
    };
    let candidate_mib = model_quant_mib.saturating_add(packed_hot_extra_mib);
    let budget_mib = requested_mib.min(available_mib).min(candidate_mib);
    let raw_quant_target_mib = budget_mib.min(model_quant_mib);
    let packed_promotion_target_mib = budget_mib.saturating_sub(raw_quant_target_mib);

    Ok(QuantResidentBudgetPlan {
        enabled: budget_mib > 0,
        raw_quant_target_mib,
        packed_promotion_target_mib,
    })
}

fn quant_resident_reserve_mib(total_mib: usize) -> usize {
    let ratio = total_mib.saturating_mul(35) / 100;
    let floor = (total_mib / 4).clamp(1024, 4096);
    ratio.max(floor)
}

#[cfg(test)]
pub fn quant_resident_budget_plan_for_test(
    total_mib: usize,
    free_mib: usize,
    model_quant_mib: usize,
    packed_hot_extra_mib: usize,
) -> Result<QuantResidentBudgetPlan, String> {
    quant_resident_budget_plan(total_mib, free_mib, model_quant_mib, packed_hot_extra_mib)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_resident_env_parser_accepts_auto_and_off() {
        assert_eq!(
            parse_quant_resident_env_value(Some("AUTO")).expect("parse AUTO"),
            QuantResidentEnv::Auto
        );
        assert_eq!(
            parse_quant_resident_env_value(Some(" off ")).expect("parse off"),
            QuantResidentEnv::Disabled
        );
        assert_eq!(
            parse_quant_resident_env_value(None).expect("parse unset"),
            QuantResidentEnv::Auto
        );
    }

    #[test]
    fn quant_resident_env_parser_rejects_invalid_values() {
        assert!(parse_quant_resident_env_value(Some("-1"))
            .expect_err("reject negative MiB")
            .contains("RNB_CUDA_QUANT_RESIDENT_MB must be auto, off, or integer MiB"));
        assert!(parse_quant_resident_env_value(Some("abc"))
            .expect_err("reject non-numeric MiB")
            .contains("RNB_CUDA_QUANT_RESIDENT_MB must be auto, off, or integer MiB"));
    }
}
