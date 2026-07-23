use std::ffi::CString;
#[cfg(windows)]
use std::os::raw::c_char;
use std::os::raw::c_void;

pub fn open_cuda() -> Result<usize, String> {
    open_first(&cuda_library_names(), "CUDA driver")
}

pub fn open_cublas() -> Result<usize, String> {
    open_first(&cublas_library_names(), "cuBLAS")
}

pub unsafe fn close(handle: usize) {
    platform_close(handle);
}

pub unsafe fn symbol<T>(lib_handle: usize, name: &str) -> Result<T, String>
where
    T: Copy,
{
    let ptr = platform_symbol(lib_handle, name)?;
    Ok(std::mem::transmute_copy(&ptr))
}

pub unsafe fn symbol_optional<T>(lib_handle: usize, name: &str) -> Option<T>
where
    T: Copy,
{
    let ptr = platform_symbol_optional(lib_handle, name)?;
    Some(std::mem::transmute_copy(&ptr))
}

fn open_first(names: &[&str], label: &str) -> Result<usize, String> {
    for name in names {
        if let Some(handle) = platform_open(name)? {
            return Ok(handle);
        }
    }
    Err(format!(
        "could not load {label} library: tried {}",
        names.join(", ")
    ))
}

#[cfg(windows)]
fn cuda_library_names() -> [&'static str; 1] {
    ["nvcuda.dll"]
}

#[cfg(not(windows))]
fn cuda_library_names() -> [&'static str; 2] {
    ["libcuda.so.1", "libcuda.so"]
}

#[cfg(windows)]
fn cublas_library_names() -> [&'static str; 2] {
    ["cublas64_12.dll", "cublas64_11.dll"]
}

#[cfg(not(windows))]
fn cublas_library_names() -> [&'static str; 2] {
    ["libcublas.so.12", "libcublas.so"]
}

#[cfg(windows)]
fn platform_open(name: &str) -> Result<Option<usize>, String> {
    let c_name = CString::new(name).map_err(|e| format!("bad library name {name}: {e}"))?;
    let handle = unsafe { LoadLibraryA(c_name.as_ptr()) };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(handle as usize))
    }
}

#[cfg(not(windows))]
fn platform_open(name: &str) -> Result<Option<usize>, String> {
    let c_name = CString::new(name).map_err(|e| format!("bad library name {name}: {e}"))?;
    let handle = unsafe { libc::dlopen(c_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        Ok(None)
    } else {
        Ok(Some(handle as usize))
    }
}

#[cfg(windows)]
unsafe fn platform_close(handle: usize) {
    let _ = FreeLibrary(handle as *mut c_void);
}

#[cfg(not(windows))]
unsafe fn platform_close(handle: usize) {
    let _ = libc::dlclose(handle as *mut libc::c_void);
}

#[cfg(windows)]
fn platform_symbol(lib_handle: usize, name: &str) -> Result<*mut c_void, String> {
    platform_symbol_optional(lib_handle, name)
        .ok_or_else(|| format!("missing CUDA driver symbol {name}"))
}

#[cfg(not(windows))]
fn platform_symbol(lib_handle: usize, name: &str) -> Result<*mut c_void, String> {
    platform_symbol_optional(lib_handle, name)
        .ok_or_else(|| format!("missing CUDA driver symbol {name}"))
}

#[cfg(windows)]
fn platform_symbol_optional(lib_handle: usize, name: &str) -> Option<*mut c_void> {
    let c_name = CString::new(name).ok()?;
    let ptr = unsafe { GetProcAddress(lib_handle as *mut c_void, c_name.as_ptr()) };
    if ptr.is_null() {
        None
    } else {
        Some(ptr as *mut c_void)
    }
}

#[cfg(not(windows))]
fn platform_symbol_optional(lib_handle: usize, name: &str) -> Option<*mut c_void> {
    let c_name = CString::new(name).ok()?;
    let ptr = unsafe { libc::dlsym(lib_handle as *mut libc::c_void, c_name.as_ptr()) };
    if ptr.is_null() {
        None
    } else {
        Some(ptr as *mut c_void)
    }
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(lp_lib_file_name: *const c_char) -> *mut c_void;
    fn GetProcAddress(h_module: *mut c_void, lp_proc_name: *const c_char) -> *mut c_void;
    fn FreeLibrary(h_lib_module: *mut c_void) -> i32;
}
