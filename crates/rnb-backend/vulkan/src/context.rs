use crate::ffi::loader::VulkanLib;
use crate::ffi::types::*;
use std::ffi::c_void;
use std::fmt;
use std::ptr;

pub(crate) struct VulkanContext {
    pub(crate) vk: VulkanLib,
    pub(crate) instance: VkInstance,
    pub(crate) device: VkDevice,
    pub(crate) queue: VkQueue,
    pub(crate) queue_family_index: u32,
    pub(crate) memory_properties: VkPhysicalDeviceMemoryProperties,
    pub(crate) max_storage_buffer_range: u64,
    pub(crate) shader_float16: bool,
}

pub struct GpuBuffer {
    pub(crate) buffer: VkBuffer,
    pub(crate) memory: VkDeviceMemory,
    pub(crate) size: VkDeviceSize,
}

impl GpuBuffer {
    /// Backing allocation size in bytes (always >= the logical payload size
    /// requested at create time — Vulkan rounds to alignment requirements).
    ///
    /// Public so cross-crate callers (e.g. rnb-runtime's fullpath wrapper)
    /// can populate `LayerWeightHandles.*_weight_size` without reaching into
    /// the private field.
    pub fn size(&self) -> VkDeviceSize {
        self.size
    }
}

unsafe fn destroy_buffer_then_free_memory(
    device: VkDevice,
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    destroy_buffer: unsafe extern "C" fn(VkDevice, VkBuffer, *const c_void),
    free_memory: unsafe extern "C" fn(VkDevice, VkDeviceMemory, *const c_void),
) {
    destroy_buffer(device, buffer, ptr::null());
    free_memory(device, memory, ptr::null());
}

#[derive(Debug)]
pub(crate) enum BufferCreateError {
    Vulkan {
        operation: &'static str,
        result: VkResult,
    },
    Other(String),
}

impl BufferCreateError {
    pub(crate) const fn is_out_of_memory(&self) -> bool {
        matches!(
            self,
            Self::Vulkan {
                result: VK_ERROR_OUT_OF_DEVICE_MEMORY | VK_ERROR_OUT_OF_HOST_MEMORY,
                ..
            }
        )
    }
}

impl fmt::Display for BufferCreateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan { operation, result } => {
                write!(formatter, "{operation} failed: {result}")
            }
            Self::Other(message) => formatter.write_str(message),
        }
    }
}

impl VulkanContext {
    pub(crate) unsafe fn new() -> Result<Self, String> {
        let vk = VulkanLib::load()?;

        let app_info = VkApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: b"rnb\0".as_ptr(),
            application_version: 1,
            p_engine_name: b"rnb-backend-vulkan\0".as_ptr(),
            engine_version: 1,
            api_version: VK_API_VERSION_1_1,
        };

        let create_info = VkInstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
        };

        let mut instance: VkInstance = ptr::null_mut();
        let res = (vk.create_instance)(&create_info, ptr::null(), &mut instance);
        if res != VK_SUCCESS {
            return Err(format!("vkCreateInstance failed: {}", res));
        }

        let mut count = 0u32;
        (vk.enumerate_physical_devices)(instance, &mut count, ptr::null_mut());
        if count == 0 {
            return Err("no Vulkan physical devices".into());
        }
        let mut devices = vec![ptr::null_mut(); count as usize];
        (vk.enumerate_physical_devices)(instance, &mut count, devices.as_mut_ptr());
        let physical_device = devices[0];
        let mut physical_properties: VkPhysicalDeviceProperties = std::mem::zeroed();
        (vk.get_physical_device_properties)(physical_device, &mut physical_properties);
        let max_storage_buffer_range = u64::from(physical_properties.max_storage_buffer_range);
        if max_storage_buffer_range == 0 {
            (vk.destroy_instance)(instance, ptr::null());
            return Err("Vulkan device reports maxStorageBufferRange=0".into());
        }

        let mut qf_count = 0u32;
        (vk.get_physical_device_queue_family_properties)(
            physical_device,
            &mut qf_count,
            ptr::null_mut(),
        );
        let mut queue_families: Vec<VkQueueFamilyProperties> = (0..qf_count as usize)
            .map(|_| VkQueueFamilyProperties {
                queue_flags: 0,
                queue_count: 0,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: [0; 3],
            })
            .collect();
        (vk.get_physical_device_queue_family_properties)(
            physical_device,
            &mut qf_count,
            queue_families.as_mut_ptr(),
        );

        let queue_family_index = queue_families
            .iter()
            .position(|qf| qf.queue_flags & VK_QUEUE_COMPUTE_BIT != 0)
            .ok_or("no compute queue family")? as u32;

        let priority = 1.0f32;
        let queue_create_info = VkDeviceQueueCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: &priority,
        };

        let mut vulkan12_features: VkPhysicalDeviceVulkan12Features = std::mem::zeroed();
        vulkan12_features.s_type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12_features.shader_float16 = 1;

        let mut device_create_info = VkDeviceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: &vulkan12_features as *const _ as *const c_void,
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
            p_enabled_features: ptr::null(),
        };

        let mut device: VkDevice = ptr::null_mut();
        let mut res = (vk.create_device)(
            physical_device,
            &device_create_info,
            ptr::null(),
            &mut device,
        );
        let shader_float16 = res == VK_SUCCESS;
        if !shader_float16 {
            device_create_info.p_next = ptr::null();
            res = (vk.create_device)(
                physical_device,
                &device_create_info,
                ptr::null(),
                &mut device,
            );
        }
        if res != VK_SUCCESS {
            return Err(format!("vkCreateDevice failed: {}", res));
        }

        let mut queue: VkQueue = ptr::null_mut();
        (vk.get_device_queue)(device, queue_family_index, 0, &mut queue);

        let mut memory_properties: VkPhysicalDeviceMemoryProperties = std::mem::zeroed();
        (vk.get_physical_device_memory_properties)(physical_device, &mut memory_properties);

        Ok(Self {
            vk,
            instance,
            device,
            queue,
            queue_family_index,
            memory_properties,
            max_storage_buffer_range,
            shader_float16,
        })
    }

    /// Get total device-local memory size in bytes.
    /// Returns the size of the largest DEVICE_LOCAL heap.
    pub(crate) fn device_local_memory_budget(&self) -> u64 {
        let mut max_size = 0u64;
        for i in 0..self.memory_properties.memory_heap_count as usize {
            let heap = &self.memory_properties.memory_heaps[i];
            if heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0 {
                max_size = max_size.max(heap.size);
            }
        }
        max_size
    }

    pub(crate) fn find_memory_type(
        &self,
        type_bits: u32,
        properties: VkFlags,
    ) -> Result<u32, String> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_bits & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & properties)
                    == properties
            {
                return Ok(i);
            }
        }
        Err("no suitable memory type".into())
    }

    pub(crate) unsafe fn create_buffer(
        &self,
        size: VkDeviceSize,
        usage: VkFlags,
        memory_properties: VkFlags,
    ) -> Result<GpuBuffer, String> {
        self.try_create_buffer(size, usage, memory_properties)
            .map_err(|error| error.to_string())
    }

    pub(crate) unsafe fn try_create_buffer(
        &self,
        size: VkDeviceSize,
        usage: VkFlags,
        memory_properties: VkFlags,
    ) -> Result<GpuBuffer, BufferCreateError> {
        let buf_info = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            size,
            usage,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };

        let mut buffer: VkBuffer = VK_NULL_HANDLE;
        let result = (self.vk.create_buffer)(self.device, &buf_info, ptr::null(), &mut buffer);
        if result != VK_SUCCESS {
            return Err(BufferCreateError::Vulkan {
                operation: "vkCreateBuffer",
                result,
            });
        }

        let mut mem_req: VkMemoryRequirements = std::mem::zeroed();
        (self.vk.get_buffer_memory_requirements)(self.device, buffer, &mut mem_req);

        let mem_type_idx = match self.find_memory_type(mem_req.memory_type_bits, memory_properties)
        {
            Ok(index) => index,
            Err(error) => {
                (self.vk.destroy_buffer)(self.device, buffer, ptr::null());
                return Err(BufferCreateError::Other(error));
            }
        };

        let alloc_info = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_req.size,
            memory_type_index: mem_type_idx,
        };

        let mut memory: VkDeviceMemory = VK_NULL_HANDLE;
        let result = (self.vk.allocate_memory)(self.device, &alloc_info, ptr::null(), &mut memory);
        if result != VK_SUCCESS {
            (self.vk.destroy_buffer)(self.device, buffer, ptr::null());
            return Err(BufferCreateError::Vulkan {
                operation: "vkAllocateMemory",
                result,
            });
        }

        let result = (self.vk.bind_buffer_memory)(self.device, buffer, memory, 0);
        if result != VK_SUCCESS {
            destroy_buffer_then_free_memory(
                self.device,
                buffer,
                memory,
                self.vk.destroy_buffer,
                self.vk.free_memory,
            );
            return Err(BufferCreateError::Vulkan {
                operation: "vkBindBufferMemory",
                result,
            });
        }

        Ok(GpuBuffer {
            buffer,
            memory,
            size,
        })
    }

    pub(crate) unsafe fn upload_to_buffer(
        &self,
        buf: &GpuBuffer,
        data: &[u8],
    ) -> Result<(), String> {
        let mut mapped: *mut c_void = ptr::null_mut();
        let res = (self.vk.map_memory)(
            self.device,
            buf.memory,
            0,
            data.len() as u64,
            0,
            &mut mapped,
        );
        if res != VK_SUCCESS {
            return Err(format!("vkMapMemory failed: {}", res));
        }
        ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());
        (self.vk.unmap_memory)(self.device, buf.memory);
        Ok(())
    }

    pub(crate) unsafe fn download_from_buffer(
        &self,
        buf: &GpuBuffer,
        out: &mut [u8],
    ) -> Result<(), String> {
        let mut mapped: *mut c_void = ptr::null_mut();
        let res =
            (self.vk.map_memory)(self.device, buf.memory, 0, out.len() as u64, 0, &mut mapped);
        if res != VK_SUCCESS {
            return Err(format!("vkMapMemory failed: {}", res));
        }
        ptr::copy_nonoverlapping(mapped as *const u8, out.as_mut_ptr(), out.len());
        (self.vk.unmap_memory)(self.device, buf.memory);
        Ok(())
    }

    /// Map a buffer persistently. Returns a raw pointer valid until unmap_buffer is called.
    /// Only works for HOST_VISIBLE buffers.
    pub(crate) unsafe fn map_buffer_persistent(&self, buf: &GpuBuffer) -> Result<*mut u8, String> {
        let mut mapped: *mut c_void = ptr::null_mut();
        let res = (self.vk.map_memory)(self.device, buf.memory, 0, buf.size, 0, &mut mapped);
        if res != VK_SUCCESS {
            return Err(format!("vkMapMemory (persistent) failed: {}", res));
        }
        Ok(mapped as *mut u8)
    }

    /// Unmap a persistently mapped buffer.
    pub(crate) unsafe fn unmap_buffer(&self, buf: &GpuBuffer) {
        (self.vk.unmap_memory)(self.device, buf.memory);
    }

    /// Copy from src buffer to dst buffer using a one-shot command buffer, then wait.
    /// Uses the provided command pool and queue. Caller is responsible for ensuring
    /// src and dst are compatible sizes.
    pub(crate) unsafe fn copy_buffer_and_wait(
        &self,
        command_pool: VkCommandPool,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        size: u64,
    ) -> Result<(), String> {
        self.copy_buffer_region_and_wait(command_pool, src, 0, dst, 0, size)
    }

    /// Copy one buffer range with a one-shot command buffer, then wait.
    pub(crate) unsafe fn copy_buffer_region_and_wait(
        &self,
        command_pool: VkCommandPool,
        src: &GpuBuffer,
        src_offset: u64,
        dst: &GpuBuffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), String> {
        let src_end = src_offset
            .checked_add(size)
            .ok_or("source buffer copy range overflow")?;
        let dst_end = dst_offset
            .checked_add(size)
            .ok_or("destination buffer copy range overflow")?;
        if src_end > src.size || dst_end > dst.size {
            return Err(format!(
                "buffer copy range exceeds allocation: src={src_offset}..{src_end}/{} dst={dst_offset}..{dst_end}/{}",
                src.size, dst.size
            ));
        }

        // Allocate a temporary command buffer
        let alloc_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cmd_buf: VkCommandBuffer = ptr::null_mut();
        let res = (self.vk.allocate_command_buffers)(self.device, &alloc_info, &mut cmd_buf);
        if res != VK_SUCCESS {
            return Err(format!(
                "vkAllocateCommandBuffers (staging) failed: {}",
                res
            ));
        }

        // Record copy command
        let begin_info = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: ptr::null(),
        };
        let res = (self.vk.begin_command_buffer)(cmd_buf, &begin_info);
        if res != VK_SUCCESS {
            (self.vk.free_command_buffers)(self.device, command_pool, 1, &cmd_buf);
            return Err(format!("vkBeginCommandBuffer (staging) failed: {}", res));
        }

        let region = VkBufferCopy {
            src_offset,
            dst_offset,
            size,
        };
        (self.vk.cmd_copy_buffer)(cmd_buf, src.buffer, dst.buffer, 1, &region);
        let res = (self.vk.end_command_buffer)(cmd_buf);
        if res != VK_SUCCESS {
            (self.vk.free_command_buffers)(self.device, command_pool, 1, &cmd_buf);
            return Err(format!("vkEndCommandBuffer (staging) failed: {}", res));
        }

        // Submit and wait
        let submit_info = VkSubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &cmd_buf,
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        };
        let res = (self.vk.queue_submit)(self.queue, 1, &submit_info, VK_NULL_HANDLE);
        if res != VK_SUCCESS {
            (self.vk.free_command_buffers)(self.device, command_pool, 1, &cmd_buf);
            return Err(format!("vkQueueSubmit (staging) failed: {}", res));
        }
        let res = (self.vk.queue_wait_idle)(self.queue);
        (self.vk.free_command_buffers)(self.device, command_pool, 1, &cmd_buf);
        if res != VK_SUCCESS {
            return Err(format!("vkQueueWaitIdle (staging) failed: {}", res));
        }

        Ok(())
    }

    pub(crate) unsafe fn destroy_buffer(&self, buf: GpuBuffer) {
        destroy_buffer_then_free_memory(
            self.device,
            buf.buffer,
            buf.memory,
            self.vk.destroy_buffer,
            self.vk.free_memory,
        );
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            (self.vk.destroy_device)(self.device, ptr::null());
            (self.vk.destroy_instance)(self.instance, ptr::null());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU8, Ordering};

    const CLEANUP_ORDER_VIOLATION: u8 = u8::MAX;
    static CLEANUP_STEP: AtomicU8 = AtomicU8::new(0);

    unsafe extern "C" fn record_destroy_buffer(
        _device: VkDevice,
        _buffer: VkBuffer,
        _allocator: *const c_void,
    ) {
        if CLEANUP_STEP
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            CLEANUP_STEP.store(CLEANUP_ORDER_VIOLATION, Ordering::SeqCst);
        }
    }

    unsafe extern "C" fn record_free_memory(
        _device: VkDevice,
        _memory: VkDeviceMemory,
        _allocator: *const c_void,
    ) {
        if CLEANUP_STEP
            .compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            CLEANUP_STEP.store(CLEANUP_ORDER_VIOLATION, Ordering::SeqCst);
        }
    }

    #[test]
    fn buffer_memory_cleanup_destroys_buffer_before_freeing_memory() {
        CLEANUP_STEP.store(0, Ordering::SeqCst);

        unsafe {
            destroy_buffer_then_free_memory(
                std::ptr::null_mut(),
                VK_NULL_HANDLE,
                VK_NULL_HANDLE,
                record_destroy_buffer,
                record_free_memory,
            );
        }

        assert_eq!(CLEANUP_STEP.load(Ordering::SeqCst), 2);
    }
}
