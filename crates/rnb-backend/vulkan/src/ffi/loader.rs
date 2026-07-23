use super::types::*;
use std::ffi::c_void;

/// dlopen'd libvulkan.so의 함수 포인터 집합
pub(crate) struct VulkanLib {
    _lib: *mut c_void,

    // Instance
    pub create_instance: unsafe extern "C" fn(
        *const VkInstanceCreateInfo,
        *const c_void,
        *mut VkInstance,
    ) -> VkResult,
    pub destroy_instance: unsafe extern "C" fn(VkInstance, *const c_void),
    pub enumerate_physical_devices:
        unsafe extern "C" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> VkResult,
    pub get_physical_device_queue_family_properties:
        unsafe extern "C" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties),
    pub get_physical_device_memory_properties:
        unsafe extern "C" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties),

    // Device
    pub create_device: unsafe extern "C" fn(
        VkPhysicalDevice,
        *const VkDeviceCreateInfo,
        *const c_void,
        *mut VkDevice,
    ) -> VkResult,
    pub destroy_device: unsafe extern "C" fn(VkDevice, *const c_void),
    pub get_device_queue: unsafe extern "C" fn(VkDevice, u32, u32, *mut VkQueue),

    // Buffer & Memory
    pub create_buffer: unsafe extern "C" fn(
        VkDevice,
        *const VkBufferCreateInfo,
        *const c_void,
        *mut VkBuffer,
    ) -> VkResult,
    pub destroy_buffer: unsafe extern "C" fn(VkDevice, VkBuffer, *const c_void),
    pub get_buffer_memory_requirements:
        unsafe extern "C" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements),
    pub allocate_memory: unsafe extern "C" fn(
        VkDevice,
        *const VkMemoryAllocateInfo,
        *const c_void,
        *mut VkDeviceMemory,
    ) -> VkResult,
    pub free_memory: unsafe extern "C" fn(VkDevice, VkDeviceMemory, *const c_void),
    pub bind_buffer_memory:
        unsafe extern "C" fn(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) -> VkResult,
    pub map_memory: unsafe extern "C" fn(
        VkDevice,
        VkDeviceMemory,
        VkDeviceSize,
        VkDeviceSize,
        VkFlags,
        *mut *mut c_void,
    ) -> VkResult,
    pub unmap_memory: unsafe extern "C" fn(VkDevice, VkDeviceMemory),

    // Descriptor
    pub create_descriptor_set_layout: unsafe extern "C" fn(
        VkDevice,
        *const VkDescriptorSetLayoutCreateInfo,
        *const c_void,
        *mut VkDescriptorSetLayout,
    ) -> VkResult,
    pub destroy_descriptor_set_layout:
        unsafe extern "C" fn(VkDevice, VkDescriptorSetLayout, *const c_void),
    pub create_descriptor_pool: unsafe extern "C" fn(
        VkDevice,
        *const VkDescriptorPoolCreateInfo,
        *const c_void,
        *mut VkDescriptorPool,
    ) -> VkResult,
    pub destroy_descriptor_pool: unsafe extern "C" fn(VkDevice, VkDescriptorPool, *const c_void),
    pub allocate_descriptor_sets: unsafe extern "C" fn(
        VkDevice,
        *const VkDescriptorSetAllocateInfo,
        *mut VkDescriptorSet,
    ) -> VkResult,
    pub update_descriptor_sets:
        unsafe extern "C" fn(VkDevice, u32, *const VkWriteDescriptorSet, u32, *const c_void),

    // Pipeline
    pub create_shader_module: unsafe extern "C" fn(
        VkDevice,
        *const VkShaderModuleCreateInfo,
        *const c_void,
        *mut VkShaderModule,
    ) -> VkResult,
    pub destroy_shader_module: unsafe extern "C" fn(VkDevice, VkShaderModule, *const c_void),
    pub create_pipeline_layout: unsafe extern "C" fn(
        VkDevice,
        *const VkPipelineLayoutCreateInfo,
        *const c_void,
        *mut VkPipelineLayout,
    ) -> VkResult,
    pub destroy_pipeline_layout: unsafe extern "C" fn(VkDevice, VkPipelineLayout, *const c_void),
    pub create_compute_pipelines: unsafe extern "C" fn(
        VkDevice,
        u64,
        u32,
        *const VkComputePipelineCreateInfo,
        *const c_void,
        *mut VkPipeline,
    ) -> VkResult,
    pub destroy_pipeline: unsafe extern "C" fn(VkDevice, VkPipeline, *const c_void),

    // Command
    pub create_command_pool: unsafe extern "C" fn(
        VkDevice,
        *const VkCommandPoolCreateInfo,
        *const c_void,
        *mut VkCommandPool,
    ) -> VkResult,
    pub destroy_command_pool: unsafe extern "C" fn(VkDevice, VkCommandPool, *const c_void),
    pub allocate_command_buffers: unsafe extern "C" fn(
        VkDevice,
        *const VkCommandBufferAllocateInfo,
        *mut VkCommandBuffer,
    ) -> VkResult,
    pub begin_command_buffer:
        unsafe extern "C" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> VkResult,
    pub end_command_buffer: unsafe extern "C" fn(VkCommandBuffer) -> VkResult,
    pub reset_command_buffer: unsafe extern "C" fn(VkCommandBuffer, VkFlags) -> VkResult,
    pub cmd_bind_pipeline: unsafe extern "C" fn(VkCommandBuffer, u32, VkPipeline),
    pub cmd_bind_descriptor_sets: unsafe extern "C" fn(
        VkCommandBuffer,
        u32,
        VkPipelineLayout,
        u32,
        u32,
        *const VkDescriptorSet,
        u32,
        *const u32,
    ),
    pub cmd_dispatch: unsafe extern "C" fn(VkCommandBuffer, u32, u32, u32),
    pub cmd_pipeline_barrier: unsafe extern "C" fn(
        VkCommandBuffer,
        VkFlags,
        VkFlags,
        VkFlags,
        u32,
        *const c_void,
        u32,
        *const VkBufferMemoryBarrier,
        u32,
        *const c_void,
    ),
    pub cmd_push_constants:
        unsafe extern "C" fn(VkCommandBuffer, VkPipelineLayout, VkFlags, u32, u32, *const c_void),
    pub cmd_copy_buffer:
        unsafe extern "C" fn(VkCommandBuffer, VkBuffer, VkBuffer, u32, *const VkBufferCopy),
    pub cmd_reset_query_pool: unsafe extern "C" fn(VkCommandBuffer, VkQueryPool, u32, u32),
    pub cmd_write_timestamp: unsafe extern "C" fn(VkCommandBuffer, VkFlags, VkQueryPool, u32),

    // Query
    pub create_query_pool: unsafe extern "C" fn(
        VkDevice,
        *const VkQueryPoolCreateInfo,
        *const c_void,
        *mut VkQueryPool,
    ) -> VkResult,
    pub destroy_query_pool: unsafe extern "C" fn(VkDevice, VkQueryPool, *const c_void),
    pub get_query_pool_results: unsafe extern "C" fn(
        VkDevice,
        VkQueryPool,
        u32,
        u32,
        usize,
        *mut c_void,
        VkDeviceSize,
        VkFlags,
    ) -> VkResult,

    // Sync
    pub queue_submit: unsafe extern "C" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> VkResult,
    pub queue_wait_idle: unsafe extern "C" fn(VkQueue) -> VkResult,
    pub create_fence: unsafe extern "C" fn(
        VkDevice,
        *const VkFenceCreateInfo,
        *const c_void,
        *mut VkFence,
    ) -> VkResult,
    pub destroy_fence: unsafe extern "C" fn(VkDevice, VkFence, *const c_void),
    pub wait_for_fences:
        unsafe extern "C" fn(VkDevice, u32, *const VkFence, VkBool32, u64) -> VkResult,
    pub reset_fences: unsafe extern "C" fn(VkDevice, u32, *const VkFence) -> VkResult,
}

extern "C" {
    fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}

const RTLD_NOW: i32 = 2;

fn vulkan_library_candidates() -> [&'static [u8]; 2] {
    [b"libvulkan.so\0", b"libvulkan.so.1\0"]
}

impl VulkanLib {
    pub(crate) unsafe fn load() -> Result<Self, String> {
        let mut lib = std::ptr::null_mut();
        for candidate in vulkan_library_candidates() {
            lib = dlopen(candidate.as_ptr(), RTLD_NOW);
            if !lib.is_null() {
                break;
            }
        }
        if lib.is_null() {
            return Err("failed to load libvulkan.so or libvulkan.so.1".into());
        }

        macro_rules! load_fn {
            ($lib:expr, $name:literal) => {{
                let sym = dlsym($lib, concat!($name, "\0").as_ptr());
                if sym.is_null() {
                    return Err(format!("missing symbol: {}", $name));
                }
                std::mem::transmute(sym)
            }};
        }

        Ok(Self {
            _lib: lib,
            create_instance: load_fn!(lib, "vkCreateInstance"),
            destroy_instance: load_fn!(lib, "vkDestroyInstance"),
            enumerate_physical_devices: load_fn!(lib, "vkEnumeratePhysicalDevices"),
            get_physical_device_queue_family_properties: load_fn!(
                lib,
                "vkGetPhysicalDeviceQueueFamilyProperties"
            ),
            get_physical_device_memory_properties: load_fn!(
                lib,
                "vkGetPhysicalDeviceMemoryProperties"
            ),
            create_device: load_fn!(lib, "vkCreateDevice"),
            destroy_device: load_fn!(lib, "vkDestroyDevice"),
            get_device_queue: load_fn!(lib, "vkGetDeviceQueue"),
            create_buffer: load_fn!(lib, "vkCreateBuffer"),
            destroy_buffer: load_fn!(lib, "vkDestroyBuffer"),
            get_buffer_memory_requirements: load_fn!(lib, "vkGetBufferMemoryRequirements"),
            allocate_memory: load_fn!(lib, "vkAllocateMemory"),
            free_memory: load_fn!(lib, "vkFreeMemory"),
            bind_buffer_memory: load_fn!(lib, "vkBindBufferMemory"),
            map_memory: load_fn!(lib, "vkMapMemory"),
            unmap_memory: load_fn!(lib, "vkUnmapMemory"),
            create_descriptor_set_layout: load_fn!(lib, "vkCreateDescriptorSetLayout"),
            destroy_descriptor_set_layout: load_fn!(lib, "vkDestroyDescriptorSetLayout"),
            create_descriptor_pool: load_fn!(lib, "vkCreateDescriptorPool"),
            destroy_descriptor_pool: load_fn!(lib, "vkDestroyDescriptorPool"),
            allocate_descriptor_sets: load_fn!(lib, "vkAllocateDescriptorSets"),
            update_descriptor_sets: load_fn!(lib, "vkUpdateDescriptorSets"),
            create_shader_module: load_fn!(lib, "vkCreateShaderModule"),
            destroy_shader_module: load_fn!(lib, "vkDestroyShaderModule"),
            create_pipeline_layout: load_fn!(lib, "vkCreatePipelineLayout"),
            destroy_pipeline_layout: load_fn!(lib, "vkDestroyPipelineLayout"),
            create_compute_pipelines: load_fn!(lib, "vkCreateComputePipelines"),
            destroy_pipeline: load_fn!(lib, "vkDestroyPipeline"),
            create_command_pool: load_fn!(lib, "vkCreateCommandPool"),
            destroy_command_pool: load_fn!(lib, "vkDestroyCommandPool"),
            allocate_command_buffers: load_fn!(lib, "vkAllocateCommandBuffers"),
            begin_command_buffer: load_fn!(lib, "vkBeginCommandBuffer"),
            end_command_buffer: load_fn!(lib, "vkEndCommandBuffer"),
            reset_command_buffer: load_fn!(lib, "vkResetCommandBuffer"),
            cmd_bind_pipeline: load_fn!(lib, "vkCmdBindPipeline"),
            cmd_bind_descriptor_sets: load_fn!(lib, "vkCmdBindDescriptorSets"),
            cmd_dispatch: load_fn!(lib, "vkCmdDispatch"),
            cmd_pipeline_barrier: load_fn!(lib, "vkCmdPipelineBarrier"),
            cmd_push_constants: load_fn!(lib, "vkCmdPushConstants"),
            cmd_copy_buffer: load_fn!(lib, "vkCmdCopyBuffer"),
            cmd_reset_query_pool: load_fn!(lib, "vkCmdResetQueryPool"),
            cmd_write_timestamp: load_fn!(lib, "vkCmdWriteTimestamp"),
            create_query_pool: load_fn!(lib, "vkCreateQueryPool"),
            destroy_query_pool: load_fn!(lib, "vkDestroyQueryPool"),
            get_query_pool_results: load_fn!(lib, "vkGetQueryPoolResults"),
            queue_submit: load_fn!(lib, "vkQueueSubmit"),
            queue_wait_idle: load_fn!(lib, "vkQueueWaitIdle"),
            create_fence: load_fn!(lib, "vkCreateFence"),
            destroy_fence: load_fn!(lib, "vkDestroyFence"),
            wait_for_fences: load_fn!(lib, "vkWaitForFences"),
            reset_fences: load_fn!(lib, "vkResetFences"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::vulkan_library_candidates;

    #[test]
    fn test_vulkan_loader_tries_versioned_soname_after_unversioned_name() {
        let candidates = vulkan_library_candidates();
        assert_eq!(candidates[0], b"libvulkan.so\0");
        assert_eq!(candidates[1], b"libvulkan.so.1\0");
    }
}

impl Drop for VulkanLib {
    fn drop(&mut self) {
        unsafe {
            dlclose(self._lib);
        }
    }
}
