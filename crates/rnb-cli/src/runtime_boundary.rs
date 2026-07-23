pub fn compiled_runtime_backends() -> Vec<&'static str> {
    rnb_llm::compiled_runtime_backends()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_runtime_backend_list_follows_features() {
        let backends = compiled_runtime_backends();

        #[cfg(feature = "cpu")]
        assert!(backends.contains(&"cpu"));
        #[cfg(feature = "cuda")]
        assert!(backends.contains(&"cuda"));
        #[cfg(feature = "vulkan")]
        assert!(backends.contains(&"vulkan"));
        #[cfg(feature = "opencl")]
        assert!(backends.contains(&"opencl"));
    }
}
