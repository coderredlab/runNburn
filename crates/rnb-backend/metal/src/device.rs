use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion,
};

/// Open the system default Metal device and return its name, or None if no
/// device is available (e.g. headless CI). No fake fallback.
pub fn system_default_device_name() -> Option<String> {
    let device: Retained<ProtocolObject<dyn MTLDevice>> = MTLCreateSystemDefaultDevice()?;
    let name: Retained<NSString> = device.name();
    Some(name.to_string())
}

/// pm34: M5 GPU neural accelerator(`mpp::tensor_ops`) 가용 여부를 결정적으로 판정한다.
/// device 는 호출측(build_metal_context)이 이미 만든 것을 받는다 — 재생성 금지.
/// (0) force override `RNB_METAL_PREFILL_FFN_KERNEL=tensorops|naive` 최우선(측정/디버그).
/// (1) `MTLGPUFamily::Apple10` 이상(Apple9=A17/M3/M4 false positive 차단) + device name "M5" cross-check.
/// (2) `mpp::tensor_ops` 최소 셰이더가 Version4_0 으로 런타임 컴파일됨(OS/SDK 가용성 흡수).
///
/// wall-time(실행 속도)은 여기서 보지 않는다(diagnostic only). 컴파일 probe 는 결정적이라
/// thermal 무관하며, 헤더가 없는 OS/SDK 면 Err → false(panic 금지).
pub fn tensorops_capability_for_device(device: &ProtocolObject<dyn MTLDevice>) -> bool {
    // (0) force override
    match std::env::var("RNB_METAL_PREFILL_FFN_KERNEL").as_deref() {
        Ok("tensorops") => return true,
        Ok("naive") => return false,
        _ => {}
    }
    // (1) family + device name cross-check
    let name = device.name().to_string();
    let family_ok = device.supportsFamily(MTLGPUFamily::Apple10) && name.contains("M5");
    // (2) compile probe (결정적): mpp::tensor_ops 최소 커널
    const PROBE_SRC: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void cap_probe(device half *a [[buffer(0)]], device half *b [[buffer(1)]],
                      device float *c [[buffer(2)]]) {
    auto A = tensor<device half, dextents<int32_t,2>, tensor_inline>(a, dextents<int32_t,2>(16, 16));
    auto B = tensor<device half, dextents<int32_t,2>, tensor_inline>(b, dextents<int32_t,2>(16, 16));
    auto C = tensor<device float, dextents<int32_t,2>, tensor_inline>(c, dextents<int32_t,2>(16, 16));
    constexpr auto d = matmul2d_descriptor(16, 16, 16, false, false, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<1>> op;
    op.run(A, B, C);
}
"#;
    let opts = MTLCompileOptions::new();
    opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let src = NSString::from_str(PROBE_SRC);
    let compile_ok = device
        .newLibraryWithSource_options_error(&src, Some(&opts))
        .is_ok();
    eprintln!(
        "[pm34][cap] family_apple10_and_m5={family_ok} name={name:?} compile_probe={compile_ok}"
    );
    family_ok && compile_ok
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires Metal device; M5 -> true, else false"]
    fn tensorops_capability_detects() {
        use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
        let device = MTLCreateSystemDefaultDevice().expect("metal device");
        let cap = super::tensorops_capability_for_device(&device);
        eprintln!(
            "[pm34] tensorops_capability = {cap} (device={:?})",
            device.name().to_string()
        );
        if device.name().to_string().contains("M5") {
            assert!(cap, "M5 device must report tensorops-capable");
        }
    }
}
