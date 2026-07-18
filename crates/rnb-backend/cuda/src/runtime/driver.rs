use std::ffi::CString;

pub(super) struct CudaApi {
    cu_init: unsafe extern "C" fn(u32) -> i32,
    cu_device_get: unsafe extern "C" fn(*mut i32, i32) -> i32,
    // cu58 pre diag: device name + PCI bus ID 로 실제 GPU 확인
    cu_device_get_count: Option<unsafe extern "C" fn(*mut i32) -> i32>,
    cu_device_get_name: Option<unsafe extern "C" fn(*mut i8, i32, i32) -> i32>,
    cu_device_get_pci_bus_id: Option<unsafe extern "C" fn(*mut i8, i32, i32) -> i32>,
    cu_ctx_create: unsafe extern "C" fn(*mut *mut libc::c_void, u32, i32) -> i32,
    cu_ctx_set_current: unsafe extern "C" fn(*mut libc::c_void) -> i32,
    cu_ctx_destroy: unsafe extern "C" fn(*mut libc::c_void) -> i32,
    cu_stream_create: unsafe extern "C" fn(*mut *mut libc::c_void, u32) -> i32,
    cu_stream_destroy: unsafe extern "C" fn(*mut libc::c_void) -> i32,
    cu_stream_synchronize: unsafe extern "C" fn(*mut libc::c_void) -> i32,
    cu_mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32,
    cu_mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    cu_mem_free: unsafe extern "C" fn(u64) -> i32,
    cu_mem_host_alloc: unsafe extern "C" fn(*mut *mut libc::c_void, usize, u32) -> i32,
    cu_mem_free_host: unsafe extern "C" fn(*mut libc::c_void) -> i32,
    cu_mem_host_register: Option<unsafe extern "C" fn(*mut libc::c_void, usize, u32) -> i32>,
    cu_mem_host_unregister: Option<unsafe extern "C" fn(*mut libc::c_void) -> i32>,
    cu_memcpy_htod_async:
        unsafe extern "C" fn(u64, *const libc::c_void, usize, *mut libc::c_void) -> i32,
    cu_memcpy_dtoh_async:
        unsafe extern "C" fn(*mut libc::c_void, u64, usize, *mut libc::c_void) -> i32,
    cu_memcpy_dtod_async: unsafe extern "C" fn(u64, u64, usize, *mut libc::c_void) -> i32,
    cu_module_load_data: unsafe extern "C" fn(*mut *mut libc::c_void, *const libc::c_void) -> i32,
    cu_module_unload: unsafe extern "C" fn(*mut libc::c_void) -> i32,
    cu_module_get_function:
        unsafe extern "C" fn(*mut *mut libc::c_void, *mut libc::c_void, *const i8) -> i32,
    cu_launch_kernel: unsafe extern "C" fn(
        *mut libc::c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut libc::c_void,
        *mut *mut libc::c_void,
        *mut *mut libc::c_void,
    ) -> i32,
    cu_stream_begin_capture: Option<unsafe extern "C" fn(*mut libc::c_void, u32) -> i32>,
    cu_stream_end_capture:
        Option<unsafe extern "C" fn(*mut libc::c_void, *mut *mut libc::c_void) -> i32>,
    cu_graph_instantiate_with_flags:
        Option<unsafe extern "C" fn(*mut *mut libc::c_void, *mut libc::c_void, u64) -> i32>,
    cu_graph_launch: Option<unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void) -> i32>,
    cu_graph_destroy: Option<unsafe extern "C" fn(*mut libc::c_void) -> i32>,
    cu_graph_exec_destroy: Option<unsafe extern "C" fn(*mut libc::c_void) -> i32>,
    // cu61 axis A: Cooperative Groups grid.sync() 지원용. regular cuLaunchKernel 은 UB.
    cu_launch_cooperative_kernel: Option<
        unsafe extern "C" fn(
            f: *mut libc::c_void,
            grid_x: u32,
            grid_y: u32,
            grid_z: u32,
            block_x: u32,
            block_y: u32,
            block_z: u32,
            shared_mem_bytes: u32,
            stream: *mut libc::c_void,
            kernel_params: *mut *mut libc::c_void,
        ) -> i32,
    >,
    cu_occupancy_max_active_blocks_per_multiprocessor: Option<
        unsafe extern "C" fn(
            num_blocks: *mut i32,
            f: *mut libc::c_void,
            block_size: i32,
            dynamic_shared_mem_bytes: usize,
        ) -> i32,
    >,
    // cu61 axis A Task 3: cooperative launch occupancy 계산 위해 SM 수 필요.
    // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16.
    cu_ctx_get_device: Option<unsafe extern "C" fn(*mut i32) -> i32>,
    cu_device_get_attribute: Option<unsafe extern "C" fn(*mut i32, i32, i32) -> i32>,
    // cu62: counter-based megakernel 의 counter zero-init 용.
    cu_memset_d32_async: unsafe extern "C" fn(u64, u32, usize, *mut libc::c_void) -> i32,
}

pub(super) const CUBLAS_OP_N: i32 = 0;
pub(super) const CUBLAS_OP_T: i32 = 1;
pub(super) const CUDA_R_32F: i32 = 0;
pub(super) const CUDA_R_16F: i32 = 2;
pub(super) const CUBLAS_COMPUTE_32F: i32 = 68;
pub(super) const CUBLAS_GEMM_DEFAULT_TENSOR_OP: i32 = 99;

// cuBLAS math mode values (cublasMath_t enum)
pub(super) const CUBLAS_DEFAULT_MATH: i32 = 0;
pub(super) const CUBLAS_PEDANTIC_MATH: i32 = 1;
pub(super) const CUBLAS_TF32_TENSOR_OP_MATH: i32 = 2;

pub(super) struct CublasApi {
    cublas_create: unsafe extern "C" fn(*mut *mut libc::c_void) -> i32,
    cublas_destroy: unsafe extern "C" fn(*mut libc::c_void) -> i32,
    cublas_set_stream: unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void) -> i32,
    cublas_set_math_mode: Option<unsafe extern "C" fn(*mut libc::c_void, i32) -> i32>,
    cublas_sgemm: unsafe extern "C" fn(
        *mut libc::c_void,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const f32,
        u64,
        i32,
        u64,
        i32,
        *const f32,
        u64,
        i32,
    ) -> i32,
    cublas_gemm_ex: Option<
        unsafe extern "C" fn(
            *mut libc::c_void,
            i32,
            i32,
            i32,
            i32,
            i32,
            *const libc::c_void,
            u64,
            i32,
            i32,
            u64,
            i32,
            i32,
            *const libc::c_void,
            u64,
            i32,
            i32,
            i32,
            i32,
        ) -> i32,
    >,
}

impl CublasApi {
    pub(super) unsafe fn load(lib_handle: usize) -> Result<Self, String> {
        Ok(Self {
            cublas_create: load_symbol(lib_handle, "cublasCreate_v2")?,
            cublas_destroy: load_symbol(lib_handle, "cublasDestroy_v2")?,
            cublas_set_stream: load_symbol(lib_handle, "cublasSetStream_v2")?,
            cublas_set_math_mode: load_symbol_optional(lib_handle, "cublasSetMathMode"),
            cublas_sgemm: load_symbol(lib_handle, "cublasSgemm_v2")?,
            cublas_gemm_ex: load_symbol_optional(lib_handle, "cublasGemmEx"),
        })
    }

    pub(super) unsafe fn set_math_mode(&self, handle: usize, mode: i32) -> Result<(), String> {
        let Some(set_mode) = self.cublas_set_math_mode else {
            return Ok(());
        };
        check_cublas(
            set_mode(handle as *mut libc::c_void, mode),
            "cublasSetMathMode",
        )
    }

    pub(super) unsafe fn create(&self) -> Result<usize, String> {
        let mut handle = std::ptr::null_mut();
        check_cublas((self.cublas_create)(&mut handle), "cublasCreate_v2")?;
        Ok(handle as usize)
    }

    pub(super) unsafe fn destroy(&self, handle: usize) -> Result<(), String> {
        check_cublas(
            (self.cublas_destroy)(handle as *mut libc::c_void),
            "cublasDestroy_v2",
        )
    }

    pub(super) unsafe fn set_stream(&self, handle: usize, stream: usize) -> Result<(), String> {
        check_cublas(
            (self.cublas_set_stream)(handle as *mut libc::c_void, stream as *mut libc::c_void),
            "cublasSetStream_v2",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn sgemm(
        &self,
        handle: usize,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: u64,
        lda: i32,
        b: u64,
        ldb: i32,
        beta: f32,
        c: u64,
        ldc: i32,
    ) -> Result<(), String> {
        check_cublas(
            (self.cublas_sgemm)(
                handle as *mut libc::c_void,
                transa,
                transb,
                m,
                n,
                k,
                &alpha as *const f32,
                a,
                lda,
                b,
                ldb,
                &beta as *const f32,
                c,
                ldc,
            ),
            "cublasSgemm_v2",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm_ex_half_half_to_f32(
        &self,
        handle: usize,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: u64,
        lda: i32,
        b: u64,
        ldb: i32,
        beta: f32,
        c: u64,
        ldc: i32,
    ) -> Result<(), String> {
        let Some(gemm_ex) = self.cublas_gemm_ex else {
            return Err("cublasGemmEx is unavailable".to_string());
        };
        check_cublas(
            gemm_ex(
                handle as *mut libc::c_void,
                transa,
                transb,
                m,
                n,
                k,
                &alpha as *const f32 as *const libc::c_void,
                a,
                CUDA_R_16F,
                lda,
                b,
                CUDA_R_16F,
                ldb,
                &beta as *const f32 as *const libc::c_void,
                c,
                CUDA_R_32F,
                ldc,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            ),
            "cublasGemmEx",
        )
    }
}

impl CudaApi {
    pub(super) unsafe fn load(lib_handle: usize) -> Result<Self, String> {
        Ok(Self {
            cu_init: load_symbol(lib_handle, "cuInit")?,
            cu_device_get: load_symbol(lib_handle, "cuDeviceGet")?,
            cu_device_get_count: load_symbol_optional(lib_handle, "cuDeviceGetCount"),
            cu_device_get_name: load_symbol_optional(lib_handle, "cuDeviceGetName"),
            cu_device_get_pci_bus_id: load_symbol_optional(lib_handle, "cuDeviceGetPCIBusId"),
            cu_ctx_create: load_symbol(lib_handle, "cuCtxCreate_v2")?,
            cu_ctx_set_current: load_symbol(lib_handle, "cuCtxSetCurrent")?,
            cu_ctx_destroy: load_symbol(lib_handle, "cuCtxDestroy_v2")?,
            cu_stream_create: load_symbol(lib_handle, "cuStreamCreate")?,
            cu_stream_destroy: load_symbol(lib_handle, "cuStreamDestroy_v2")?,
            cu_stream_synchronize: load_symbol(lib_handle, "cuStreamSynchronize")?,
            cu_mem_get_info: load_symbol(lib_handle, "cuMemGetInfo_v2")?,
            cu_mem_alloc: load_symbol(lib_handle, "cuMemAlloc_v2")?,
            cu_mem_free: load_symbol(lib_handle, "cuMemFree_v2")?,
            cu_mem_host_alloc: load_symbol(lib_handle, "cuMemHostAlloc")?,
            cu_mem_free_host: load_symbol(lib_handle, "cuMemFreeHost")?,
            cu_mem_host_register: load_symbol_optional(lib_handle, "cuMemHostRegister_v2")
                .or_else(|| load_symbol_optional(lib_handle, "cuMemHostRegister")),
            cu_mem_host_unregister: load_symbol_optional(lib_handle, "cuMemHostUnregister"),
            cu_memcpy_htod_async: load_symbol(lib_handle, "cuMemcpyHtoDAsync_v2")?,
            cu_memcpy_dtoh_async: load_symbol(lib_handle, "cuMemcpyDtoHAsync_v2")?,
            cu_memcpy_dtod_async: load_symbol(lib_handle, "cuMemcpyDtoDAsync_v2")?,
            cu_module_load_data: load_symbol(lib_handle, "cuModuleLoadData")?,
            cu_module_unload: load_symbol(lib_handle, "cuModuleUnload")?,
            cu_module_get_function: load_symbol(lib_handle, "cuModuleGetFunction")?,
            cu_launch_kernel: load_symbol(lib_handle, "cuLaunchKernel")?,
            cu_stream_begin_capture: load_symbol_optional(lib_handle, "cuStreamBeginCapture_v2"),
            cu_stream_end_capture: load_symbol_optional(lib_handle, "cuStreamEndCapture"),
            cu_graph_instantiate_with_flags: load_symbol_optional(
                lib_handle,
                "cuGraphInstantiateWithFlags",
            ),
            cu_graph_launch: load_symbol_optional(lib_handle, "cuGraphLaunch"),
            cu_graph_destroy: load_symbol_optional(lib_handle, "cuGraphDestroy"),
            cu_graph_exec_destroy: load_symbol_optional(lib_handle, "cuGraphExecDestroy"),
            cu_launch_cooperative_kernel: load_symbol_optional(
                lib_handle,
                "cuLaunchCooperativeKernel",
            ),
            cu_occupancy_max_active_blocks_per_multiprocessor: load_symbol_optional(
                lib_handle,
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            ),
            cu_ctx_get_device: load_symbol_optional(lib_handle, "cuCtxGetDevice"),
            cu_device_get_attribute: load_symbol_optional(lib_handle, "cuDeviceGetAttribute"),
            cu_memset_d32_async: load_symbol(lib_handle, "cuMemsetD32Async")?,
        })
    }

    pub(super) unsafe fn init(&self, flags: u32) -> Result<(), String> {
        check_cuda((self.cu_init)(flags), "cuInit")
    }

    pub(super) unsafe fn device_get(&self, ordinal: i32) -> Result<i32, String> {
        let mut device = 0;
        check_cuda((self.cu_device_get)(&mut device, ordinal), "cuDeviceGet")?;
        Ok(device)
    }

    pub(super) unsafe fn device_get_count(&self) -> Result<i32, String> {
        let f = self
            .cu_device_get_count
            .ok_or_else(|| "cuDeviceGetCount unavailable".to_string())?;
        let mut count = 0i32;
        check_cuda(f(&mut count), "cuDeviceGetCount")?;
        Ok(count)
    }

    pub(super) unsafe fn device_get_name(&self, device: i32) -> Result<String, String> {
        let f = self
            .cu_device_get_name
            .ok_or_else(|| "cuDeviceGetName unavailable".to_string())?;
        let mut buf = [0i8; 256];
        check_cuda(
            f(buf.as_mut_ptr(), buf.len() as i32 - 1, device),
            "cuDeviceGetName",
        )?;
        Ok(std::ffi::CStr::from_ptr(buf.as_ptr())
            .to_string_lossy()
            .into_owned())
    }

    pub(super) unsafe fn device_get_pci_bus_id(&self, device: i32) -> Result<String, String> {
        let f = self
            .cu_device_get_pci_bus_id
            .ok_or_else(|| "cuDeviceGetPCIBusId unavailable".to_string())?;
        let mut buf = [0i8; 64];
        check_cuda(
            f(buf.as_mut_ptr(), buf.len() as i32 - 1, device),
            "cuDeviceGetPCIBusId",
        )?;
        Ok(std::ffi::CStr::from_ptr(buf.as_ptr())
            .to_string_lossy()
            .into_owned())
    }

    pub(super) unsafe fn ctx_create(&self, flags: u32, device: i32) -> Result<usize, String> {
        let mut ctx = std::ptr::null_mut();
        check_cuda(
            (self.cu_ctx_create)(&mut ctx, flags, device),
            "cuCtxCreate_v2",
        )?;
        Ok(ctx as usize)
    }

    pub(super) unsafe fn ctx_destroy(&self, ctx: usize) -> Result<(), String> {
        check_cuda(
            (self.cu_ctx_destroy)(ctx as *mut libc::c_void),
            "cuCtxDestroy_v2",
        )
    }

    pub(super) unsafe fn ctx_set_current(&self, ctx: usize) -> Result<(), String> {
        check_cuda(
            (self.cu_ctx_set_current)(ctx as *mut libc::c_void),
            "cuCtxSetCurrent",
        )
    }

    pub(super) unsafe fn stream_create(&self, flags: u32) -> Result<usize, String> {
        let mut stream = std::ptr::null_mut();
        check_cuda(
            (self.cu_stream_create)(&mut stream, flags),
            "cuStreamCreate",
        )?;
        Ok(stream as usize)
    }

    pub(super) unsafe fn stream_destroy(&self, stream: usize) -> Result<(), String> {
        check_cuda(
            (self.cu_stream_destroy)(stream as *mut libc::c_void),
            "cuStreamDestroy_v2",
        )
    }

    pub(super) unsafe fn stream_synchronize(&self, stream: usize) -> Result<(), String> {
        check_cuda(
            (self.cu_stream_synchronize)(stream as *mut libc::c_void),
            "cuStreamSynchronize",
        )
    }

    pub(super) unsafe fn mem_get_info(&self) -> Result<(usize, usize), String> {
        let mut free = 0usize;
        let mut total = 0usize;
        check_cuda(
            (self.cu_mem_get_info)(&mut free, &mut total),
            "cuMemGetInfo_v2",
        )?;
        Ok((free, total))
    }

    pub(super) unsafe fn mem_alloc(&self, bytes: usize) -> Result<u64, String> {
        let mut ptr = 0u64;
        check_cuda((self.cu_mem_alloc)(&mut ptr, bytes), "cuMemAlloc_v2")?;
        Ok(ptr)
    }

    pub(super) unsafe fn mem_free(&self, ptr: u64) -> Result<(), String> {
        check_cuda((self.cu_mem_free)(ptr), "cuMemFree_v2")
    }

    pub(super) unsafe fn mem_host_alloc(&self, bytes: usize) -> Result<*mut libc::c_void, String> {
        let mut ptr = std::ptr::null_mut();
        check_cuda(
            (self.cu_mem_host_alloc)(&mut ptr, bytes, 0),
            "cuMemHostAlloc",
        )?;
        Ok(ptr)
    }

    pub(super) unsafe fn mem_free_host(&self, ptr: *mut libc::c_void) -> Result<(), String> {
        check_cuda((self.cu_mem_free_host)(ptr), "cuMemFreeHost")
    }

    pub(super) unsafe fn mem_host_register(
        &self,
        ptr: *mut libc::c_void,
        bytes: usize,
        flags: u32,
    ) -> Result<(), String> {
        let Some(register) = self.cu_mem_host_register else {
            return Err("missing CUDA driver symbol cuMemHostRegister".to_string());
        };
        check_cuda(register(ptr, bytes, flags), "cuMemHostRegister")
    }

    pub(super) unsafe fn mem_host_unregister(&self, ptr: *mut libc::c_void) -> Result<(), String> {
        let Some(unregister) = self.cu_mem_host_unregister else {
            return Err("missing CUDA driver symbol cuMemHostUnregister".to_string());
        };
        check_cuda(unregister(ptr), "cuMemHostUnregister")
    }

    pub(super) unsafe fn memcpy_htod_async(
        &self,
        dst: u64,
        src: *const libc::c_void,
        bytes: usize,
        stream: usize,
    ) -> Result<(), String> {
        // cu19: trace large H2D transfers with a short backtrace so we can map
        // the 10.96 GB/prefill bulk back to the Rust caller. Threshold and toggle
        // are env-controlled to keep release runs silent.
        if bytes >= 1024 * 1024 && std::env::var("RNB_CUDA_H2D_TRACE").ok().as_deref() == Some("1")
        {
            let bt = std::backtrace::Backtrace::force_capture();
            let bt_str = format!("{bt}");
            let mut frames: Vec<&str> = bt_str
                .lines()
                .filter_map(|line| {
                    let trimmed = line.trim_start();
                    if trimmed.contains("rnb_") || trimmed.contains("crates/rnb-") {
                        Some(trimmed)
                    } else {
                        None
                    }
                })
                .take(4)
                .collect();
            frames.retain(|s| !s.contains("memcpy_htod_async") && !s.contains("Backtrace::"));
            eprintln!(
                "[h2d-trace] bytes={} MB={:.2} top={:?}",
                bytes,
                bytes as f64 / (1024.0 * 1024.0),
                frames
            );
        }
        check_cuda(
            (self.cu_memcpy_htod_async)(dst, src, bytes, stream as *mut libc::c_void),
            "cuMemcpyHtoDAsync_v2",
        )
    }

    pub(super) unsafe fn memcpy_dtoh_async(
        &self,
        dst: *mut libc::c_void,
        src: u64,
        bytes: usize,
        stream: usize,
    ) -> Result<(), String> {
        check_cuda(
            (self.cu_memcpy_dtoh_async)(dst, src, bytes, stream as *mut libc::c_void),
            "cuMemcpyDtoHAsync_v2",
        )
    }

    pub(super) unsafe fn memcpy_dtod_async(
        &self,
        dst: u64,
        src: u64,
        bytes: usize,
        stream: usize,
    ) -> Result<(), String> {
        check_cuda(
            (self.cu_memcpy_dtod_async)(dst, src, bytes, stream as *mut libc::c_void),
            "cuMemcpyDtoDAsync_v2",
        )
    }

    pub(super) unsafe fn module_load_data(&self, ptx: &str) -> Result<*mut libc::c_void, String> {
        let c_ptx = CString::new(ptx).map_err(|e| format!("bad PTX string: {e}"))?;
        let mut module = std::ptr::null_mut();
        check_cuda(
            (self.cu_module_load_data)(&mut module, c_ptx.as_ptr().cast::<libc::c_void>()),
            "cuModuleLoadData",
        )?;
        Ok(module)
    }

    pub(super) unsafe fn module_unload(&self, module: *mut libc::c_void) -> Result<(), String> {
        check_cuda((self.cu_module_unload)(module), "cuModuleUnload")
    }

    pub(super) unsafe fn module_get_function(
        &self,
        module: *mut libc::c_void,
        name: &str,
    ) -> Result<*mut libc::c_void, String> {
        let c_name = CString::new(name).map_err(|e| format!("bad kernel name {name}: {e}"))?;
        let mut function = std::ptr::null_mut();
        check_cuda(
            (self.cu_module_get_function)(&mut function, module, c_name.as_ptr()),
            "cuModuleGetFunction",
        )?;
        Ok(function)
    }

    pub(super) unsafe fn launch_kernel(
        &self,
        function: *mut libc::c_void,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: usize,
        params: *mut *mut libc::c_void,
    ) -> Result<(), String> {
        check_cuda(
            (self.cu_launch_kernel)(
                function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                stream as *mut libc::c_void,
                params,
                std::ptr::null_mut(),
            ),
            "cuLaunchKernel",
        )
    }

    // cu62: counter buffer zero-init.
    pub(super) unsafe fn memset_d32_async(
        &self,
        dst: u64,
        value: u32,
        count: usize,
        stream: usize,
    ) -> Result<(), String> {
        check_cuda(
            (self.cu_memset_d32_async)(dst, value, count, stream as *mut libc::c_void),
            "cuMemsetD32Async",
        )
    }

    // cu61 axis A: Cooperative Groups grid.sync() 가 필요한 kernel 은 반드시
    // 이 path 로 launch 해야 함. regular cuLaunchKernel 로 launch 하면 UB.
    pub(super) unsafe fn launch_cooperative_kernel(
        &self,
        function: *mut libc::c_void,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: usize,
        params: *mut *mut libc::c_void,
    ) -> Result<(), String> {
        let func = self.cu_launch_cooperative_kernel.ok_or_else(|| {
            "cuLaunchCooperativeKernel symbol not loaded (CUDA < 9 or driver too old)".to_string()
        })?;
        check_cuda(
            func(
                function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                stream as *mut libc::c_void,
                params,
            ),
            "cuLaunchCooperativeKernel",
        )
    }

    // cu61 axis A: occupancy probe — Cooperative launch 는 SM 전체에 한 wave 로
    // 들어가야 grid.sync() 가 deadlock 없이 동작하므로, 호출자가 block count 를
    // SM 수 × max-active-blocks-per-SM 이하로 clamp 할 수 있게 정보 제공.
    pub(super) unsafe fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        function: *mut libc::c_void,
        block_size: i32,
        dynamic_shared_mem_bytes: usize,
    ) -> Result<i32, String> {
        let func = self
            .cu_occupancy_max_active_blocks_per_multiprocessor
            .ok_or_else(|| {
                "cuOccupancyMaxActiveBlocksPerMultiprocessor symbol not loaded".to_string()
            })?;
        let mut num_blocks: i32 = 0;
        check_cuda(
            func(
                &mut num_blocks,
                function,
                block_size,
                dynamic_shared_mem_bytes,
            ),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )?;
        Ok(num_blocks)
    }

    // cu61 axis A Task 3: cooperative launch 의 max grid = SM count × max_blocks_per_SM.
    // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16.
    pub(super) unsafe fn device_multiprocessor_count(&self) -> Result<i32, String> {
        let get_dev = self
            .cu_ctx_get_device
            .ok_or_else(|| "cuCtxGetDevice symbol not loaded".to_string())?;
        let get_attr = self
            .cu_device_get_attribute
            .ok_or_else(|| "cuDeviceGetAttribute symbol not loaded".to_string())?;
        let mut device: i32 = 0;
        check_cuda(get_dev(&mut device), "cuCtxGetDevice")?;
        let mut count: i32 = 0;
        // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
        check_cuda(get_attr(&mut count, 16, device), "cuDeviceGetAttribute")?;
        Ok(count)
    }

    pub(super) unsafe fn stream_begin_capture(&self, stream: usize) -> Result<(), String> {
        let Some(begin_capture) = self.cu_stream_begin_capture else {
            return Err("missing CUDA driver symbol cuStreamBeginCapture_v2".to_string());
        };
        check_cuda(
            begin_capture(stream as *mut libc::c_void, 0),
            "cuStreamBeginCapture_v2",
        )
    }

    pub(super) unsafe fn stream_end_capture(
        &self,
        stream: usize,
    ) -> Result<*mut libc::c_void, String> {
        let Some(end_capture) = self.cu_stream_end_capture else {
            return Err("missing CUDA driver symbol cuStreamEndCapture".to_string());
        };
        let mut graph = std::ptr::null_mut();
        check_cuda(
            end_capture(stream as *mut libc::c_void, &mut graph),
            "cuStreamEndCapture",
        )?;
        Ok(graph)
    }

    pub(super) unsafe fn graph_instantiate(
        &self,
        graph: *mut libc::c_void,
    ) -> Result<*mut libc::c_void, String> {
        let Some(instantiate) = self.cu_graph_instantiate_with_flags else {
            return Err("missing CUDA driver symbol cuGraphInstantiateWithFlags".to_string());
        };
        let mut exec = std::ptr::null_mut();
        check_cuda(
            instantiate(&mut exec, graph, 0),
            "cuGraphInstantiateWithFlags",
        )?;
        Ok(exec)
    }

    pub(super) unsafe fn graph_launch(
        &self,
        exec: *mut libc::c_void,
        stream: usize,
    ) -> Result<(), String> {
        let Some(launch) = self.cu_graph_launch else {
            return Err("missing CUDA driver symbol cuGraphLaunch".to_string());
        };
        check_cuda(launch(exec, stream as *mut libc::c_void), "cuGraphLaunch")
    }

    pub(super) unsafe fn graph_destroy(&self, graph: *mut libc::c_void) -> Result<(), String> {
        let Some(destroy) = self.cu_graph_destroy else {
            return Err("missing CUDA driver symbol cuGraphDestroy".to_string());
        };
        check_cuda(destroy(graph), "cuGraphDestroy")
    }

    pub(super) unsafe fn graph_exec_destroy(&self, exec: *mut libc::c_void) -> Result<(), String> {
        let Some(destroy) = self.cu_graph_exec_destroy else {
            return Err("missing CUDA driver symbol cuGraphExecDestroy".to_string());
        };
        check_cuda(destroy(exec), "cuGraphExecDestroy")
    }
}

#[cfg(test)]
pub(super) const SMOKE_ADD_ONE_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry rnb_smoke_add_one(
.param .u64 out_ptr,
.param .u64 in_ptr
)
{
.reg .b64 %rd<3>;
.reg .f32 %f<2>;

ld.param.u64 %rd1, [out_ptr];
ld.param.u64 %rd2, [in_ptr];
ld.global.f32 %f1, [%rd2];
add.rn.f32 %f1, %f1, 0f3f800000;
st.global.f32 [%rd1], %f1;
ret;
}
"#;

#[cfg(test)]
pub(super) const Q4K_BLOCK_DOT_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry rnb_q4k_block_dot(
.param .u64 out_ptr,
.param .u64 block_ptr,
.param .u64 input_ptr
)
{
.reg .pred %p<2>;
.reg .b16 %h<3>;
.reg .b32 %r<18>;
.reg .b64 %rd<10>;
.reg .f32 %f<12>;

ld.param.u64 %rd1, [out_ptr];
ld.param.u64 %rd2, [block_ptr];
ld.param.u64 %rd3, [input_ptr];

ld.u16 %h1, [%rd2];
ld.u16 %h2, [%rd2+2];
cvt.f32.f16 %f1, %h1;
cvt.f32.f16 %f2, %h2;
mov.f32 %f3, 0f00000000;
mov.u32 %r1, 0;

LOOP_I:
setp.ge.u32 %p1, %r1, 256;
@%p1 bra DONE;

shr.u32 %r2, %r1, 5;
shr.u32 %r3, %r1, 6;
and.b32 %r4, %r2, 1;
shl.b32 %r5, %r3, 1;
add.u32 %r6, %r5, %r4;

setp.lt.u32 %p1, %r6, 4;
@%p1 bra SCALE_LOW;

add.u32 %r7, %r6, 4;
cvt.u64.u32 %rd7, %r7;
add.u64 %rd4, %rd2, %rd7;
ld.u8 %r8, [%rd4+4];
and.b32 %r9, %r8, 15;
sub.u32 %r10, %r6, 4;
cvt.u64.u32 %rd7, %r10;
add.u64 %rd5, %rd2, %rd7;
ld.u8 %r11, [%rd5+4];
shr.u32 %r12, %r11, 6;
shl.b32 %r12, %r12, 4;
or.b32 %r13, %r9, %r12;

shr.u32 %r9, %r8, 4;
cvt.u64.u32 %rd7, %r6;
add.u64 %rd5, %rd2, %rd7;
ld.u8 %r11, [%rd5+4];
shr.u32 %r12, %r11, 6;
shl.b32 %r12, %r12, 4;
or.b32 %r14, %r9, %r12;
bra SCALE_DONE;

SCALE_LOW:
cvt.u64.u32 %rd7, %r6;
add.u64 %rd4, %rd2, %rd7;
ld.u8 %r8, [%rd4+4];
and.b32 %r13, %r8, 63;
add.u32 %r7, %r6, 4;
cvt.u64.u32 %rd7, %r7;
add.u64 %rd5, %rd2, %rd7;
ld.u8 %r8, [%rd5+4];
and.b32 %r14, %r8, 63;

SCALE_DONE:
rem.u32 %r15, %r1, 64;
rem.u32 %r16, %r1, 32;
mul.lo.u32 %r17, %r3, 32;
add.u32 %r17, %r17, %r16;
cvt.u64.u32 %rd7, %r17;
add.u64 %rd4, %rd2, %rd7;
ld.u8 %r8, [%rd4+16];
setp.lt.u32 %p1, %r15, 32;
@%p1 bra Q_LOW;
shr.u32 %r8, %r8, 4;
bra Q_DONE;
Q_LOW:
and.b32 %r8, %r8, 15;
Q_DONE:
cvt.rn.f32.u32 %f4, %r13;
cvt.rn.f32.u32 %f5, %r14;
cvt.rn.f32.u32 %f6, %r8;
mul.rn.f32 %f7, %f1, %f4;
mul.rn.f32 %f7, %f7, %f6;
mul.rn.f32 %f8, %f2, %f5;
sub.rn.f32 %f9, %f7, %f8;
mul.wide.u32 %rd5, %r1, 4;
add.u64 %rd6, %rd3, %rd5;
ld.global.f32 %f10, [%rd6];
fma.rn.f32 %f3, %f9, %f10, %f3;
add.u32 %r1, %r1, 1;
bra LOOP_I;

DONE:
st.global.f32 [%rd1], %f3;
ret;
}
"#;

#[cfg(test)]
pub(super) const Q4K_ROW_DOT_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry rnb_q4k_row_dot(
.param .u64 out_ptr,
.param .u64 row_ptr,
.param .u64 input_ptr,
.param .u32 blocks
)
{
.reg .pred %p<3>;
.reg .b16 %h<3>;
.reg .b32 %r<24>;
.reg .b64 %rd<14>;
.reg .f32 %f<12>;

ld.param.u64 %rd1, [out_ptr];
ld.param.u64 %rd2, [row_ptr];
ld.param.u64 %rd3, [input_ptr];
ld.param.u32 %r20, [blocks];

mov.f32 %f3, 0f00000000;
mov.u32 %r21, 0;

LOOP_BLOCK:
setp.ge.u32 %p2, %r21, %r20;
@%p2 bra DONE;

mul.wide.u32 %rd8, %r21, 144;
add.u64 %rd9, %rd2, %rd8;
mul.wide.u32 %rd10, %r21, 1024;
add.u64 %rd11, %rd3, %rd10;

ld.u16 %h1, [%rd9];
ld.u16 %h2, [%rd9+2];
cvt.f32.f16 %f1, %h1;
cvt.f32.f16 %f2, %h2;
mov.u32 %r1, 0;

LOOP_I:
setp.ge.u32 %p1, %r1, 256;
@%p1 bra NEXT_BLOCK;

shr.u32 %r2, %r1, 5;
shr.u32 %r3, %r1, 6;
and.b32 %r4, %r2, 1;
shl.b32 %r5, %r3, 1;
add.u32 %r6, %r5, %r4;

setp.lt.u32 %p1, %r6, 4;
@%p1 bra SCALE_LOW;

add.u32 %r7, %r6, 4;
cvt.u64.u32 %rd7, %r7;
add.u64 %rd4, %rd9, %rd7;
ld.u8 %r8, [%rd4+4];
and.b32 %r9, %r8, 15;
sub.u32 %r10, %r6, 4;
cvt.u64.u32 %rd7, %r10;
add.u64 %rd5, %rd9, %rd7;
ld.u8 %r11, [%rd5+4];
shr.u32 %r12, %r11, 6;
shl.b32 %r12, %r12, 4;
or.b32 %r13, %r9, %r12;

shr.u32 %r9, %r8, 4;
cvt.u64.u32 %rd7, %r6;
add.u64 %rd5, %rd9, %rd7;
ld.u8 %r11, [%rd5+4];
shr.u32 %r12, %r11, 6;
shl.b32 %r12, %r12, 4;
or.b32 %r14, %r9, %r12;
bra SCALE_DONE;

SCALE_LOW:
cvt.u64.u32 %rd7, %r6;
add.u64 %rd4, %rd9, %rd7;
ld.u8 %r8, [%rd4+4];
and.b32 %r13, %r8, 63;
add.u32 %r7, %r6, 4;
cvt.u64.u32 %rd7, %r7;
add.u64 %rd5, %rd9, %rd7;
ld.u8 %r8, [%rd5+4];
and.b32 %r14, %r8, 63;

SCALE_DONE:
rem.u32 %r15, %r1, 64;
rem.u32 %r16, %r1, 32;
mul.lo.u32 %r17, %r3, 32;
add.u32 %r17, %r17, %r16;
cvt.u64.u32 %rd7, %r17;
add.u64 %rd4, %rd9, %rd7;
ld.u8 %r8, [%rd4+16];
setp.lt.u32 %p1, %r15, 32;
@%p1 bra Q_LOW;
shr.u32 %r8, %r8, 4;
bra Q_DONE;
Q_LOW:
and.b32 %r8, %r8, 15;
Q_DONE:
cvt.rn.f32.u32 %f4, %r13;
cvt.rn.f32.u32 %f5, %r14;
cvt.rn.f32.u32 %f6, %r8;
mul.rn.f32 %f7, %f1, %f4;
mul.rn.f32 %f7, %f7, %f6;
mul.rn.f32 %f8, %f2, %f5;
sub.rn.f32 %f9, %f7, %f8;
mul.wide.u32 %rd5, %r1, 4;
add.u64 %rd6, %rd11, %rd5;
ld.global.f32 %f10, [%rd6];
fma.rn.f32 %f3, %f9, %f10, %f3;
add.u32 %r1, %r1, 1;
bra LOOP_I;

NEXT_BLOCK:
add.u32 %r21, %r21, 1;
bra LOOP_BLOCK;

DONE:
st.global.f32 [%rd1], %f3;
ret;
}
"#;

pub(super) const Q4K_GEMV_PARALLEL_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/q4k_gemv.ptx"));

pub(super) const NEMOTRON_SELECTED_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/nemotron_selected.ptx"));

pub(super) const PERSISTENT_DECODE_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/persistent_decode.ptx"));

#[allow(dead_code)]
pub(super) const Q4K_GEMV_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry rnb_q4k_gemv(
.param .u64 out_ptr,
.param .u64 weights_ptr,
.param .u64 input_ptr,
.param .u32 rows,
.param .u32 blocks_per_row
)
{
.reg .pred %p<4>;
.reg .b16 %h<3>;
.reg .b32 %r<28>;
.reg .b64 %rd<18>;
.reg .f32 %f<12>;

ld.param.u64 %rd1, [out_ptr];
ld.param.u64 %rd2, [weights_ptr];
ld.param.u64 %rd3, [input_ptr];
ld.param.u32 %r20, [rows];
ld.param.u32 %r22, [blocks_per_row];

mov.u32 %r23, %ctaid.x;
setp.ge.u32 %p3, %r23, %r20;
@%p3 bra DONE_RET;

mul.lo.u32 %r24, %r22, 144;
mul.wide.u32 %rd12, %r23, %r24;
add.u64 %rd13, %rd2, %rd12;

mov.f32 %f3, 0f00000000;
mov.u32 %r21, 0;

LOOP_BLOCK:
setp.ge.u32 %p2, %r21, %r22;
@%p2 bra DONE;

mul.wide.u32 %rd8, %r21, 144;
add.u64 %rd9, %rd13, %rd8;
mul.wide.u32 %rd10, %r21, 1024;
add.u64 %rd11, %rd3, %rd10;

ld.u16 %h1, [%rd9];
ld.u16 %h2, [%rd9+2];
cvt.f32.f16 %f1, %h1;
cvt.f32.f16 %f2, %h2;
mov.u32 %r1, 0;

LOOP_I:
setp.ge.u32 %p1, %r1, 256;
@%p1 bra NEXT_BLOCK;

shr.u32 %r2, %r1, 5;
shr.u32 %r3, %r1, 6;
and.b32 %r4, %r2, 1;
shl.b32 %r5, %r3, 1;
add.u32 %r6, %r5, %r4;

setp.lt.u32 %p1, %r6, 4;
@%p1 bra SCALE_LOW;

add.u32 %r7, %r6, 4;
cvt.u64.u32 %rd7, %r7;
add.u64 %rd4, %rd9, %rd7;
ld.u8 %r8, [%rd4+4];
and.b32 %r9, %r8, 15;
sub.u32 %r10, %r6, 4;
cvt.u64.u32 %rd7, %r10;
add.u64 %rd5, %rd9, %rd7;
ld.u8 %r11, [%rd5+4];
shr.u32 %r12, %r11, 6;
shl.b32 %r12, %r12, 4;
or.b32 %r13, %r9, %r12;

shr.u32 %r9, %r8, 4;
cvt.u64.u32 %rd7, %r6;
add.u64 %rd5, %rd9, %rd7;
ld.u8 %r11, [%rd5+4];
shr.u32 %r12, %r11, 6;
shl.b32 %r12, %r12, 4;
or.b32 %r14, %r9, %r12;
bra SCALE_DONE;

SCALE_LOW:
cvt.u64.u32 %rd7, %r6;
add.u64 %rd4, %rd9, %rd7;
ld.u8 %r8, [%rd4+4];
and.b32 %r13, %r8, 63;
add.u32 %r7, %r6, 4;
cvt.u64.u32 %rd7, %r7;
add.u64 %rd5, %rd9, %rd7;
ld.u8 %r8, [%rd5+4];
and.b32 %r14, %r8, 63;

SCALE_DONE:
rem.u32 %r15, %r1, 64;
rem.u32 %r16, %r1, 32;
mul.lo.u32 %r17, %r3, 32;
add.u32 %r17, %r17, %r16;
cvt.u64.u32 %rd7, %r17;
add.u64 %rd4, %rd9, %rd7;
ld.u8 %r8, [%rd4+16];
setp.lt.u32 %p1, %r15, 32;
@%p1 bra Q_LOW;
shr.u32 %r8, %r8, 4;
bra Q_DONE;
Q_LOW:
and.b32 %r8, %r8, 15;
Q_DONE:
cvt.rn.f32.u32 %f4, %r13;
cvt.rn.f32.u32 %f5, %r14;
cvt.rn.f32.u32 %f6, %r8;
mul.rn.f32 %f7, %f1, %f4;
mul.rn.f32 %f7, %f7, %f6;
mul.rn.f32 %f8, %f2, %f5;
sub.rn.f32 %f9, %f7, %f8;
mul.wide.u32 %rd5, %r1, 4;
add.u64 %rd6, %rd11, %rd5;
ld.global.f32 %f10, [%rd6];
fma.rn.f32 %f3, %f9, %f10, %f3;
add.u32 %r1, %r1, 1;
bra LOOP_I;

NEXT_BLOCK:
add.u32 %r21, %r21, 1;
bra LOOP_BLOCK;

DONE:
mul.wide.u32 %rd14, %r23, 4;
add.u64 %rd15, %rd1, %rd14;
st.global.f32 [%rd15], %f3;
DONE_RET:
ret;
}
"#;

pub(super) fn dlopen_cuda() -> Result<usize, String> {
    crate::dynlib::open_cuda()
}

pub(super) fn dlopen_cublas() -> Result<usize, String> {
    crate::dynlib::open_cublas()
}

pub(super) unsafe fn load_symbol<T>(lib_handle: usize, name: &str) -> Result<T, String>
where
    T: Copy,
{
    crate::dynlib::symbol(lib_handle, name)
}

pub(super) unsafe fn load_symbol_optional<T>(lib_handle: usize, name: &str) -> Option<T>
where
    T: Copy,
{
    crate::dynlib::symbol_optional(lib_handle, name)
}

pub(super) fn check_cuda(code: i32, label: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{label} failed with CUDA error {code}"))
    }
}

pub(super) fn check_cublas(code: i32, label: &str) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{label} failed with cuBLAS error {code}"))
    }
}
