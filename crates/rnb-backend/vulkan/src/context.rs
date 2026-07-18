use crate::ffi::loader::VulkanLib;
use crate::ffi::types::*;
use std::ffi::c_void;
use std::ptr;

pub(crate) struct VulkanContext {
    pub(crate) vk: VulkanLib,
    pub(crate) instance: VkInstance,
    pub(crate) device: VkDevice,
    pub(crate) queue: VkQueue,
    pub(crate) queue_family_index: u32,
    pub(crate) memory_properties: VkPhysicalDeviceMemoryProperties,
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

        let device_create_info = VkDeviceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: ptr::null(),
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
        let res = (vk.create_device)(
            physical_device,
            &device_create_info,
            ptr::null(),
            &mut device,
        );
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
        let res = (self.vk.create_buffer)(self.device, &buf_info, ptr::null(), &mut buffer);
        if res != VK_SUCCESS {
            return Err(format!("vkCreateBuffer failed: {}", res));
        }

        let mut mem_req: VkMemoryRequirements = std::mem::zeroed();
        (self.vk.get_buffer_memory_requirements)(self.device, buffer, &mut mem_req);

        let mem_type_idx = self.find_memory_type(mem_req.memory_type_bits, memory_properties)?;

        let alloc_info = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_req.size,
            memory_type_index: mem_type_idx,
        };

        let mut memory: VkDeviceMemory = VK_NULL_HANDLE;
        let res = (self.vk.allocate_memory)(self.device, &alloc_info, ptr::null(), &mut memory);
        if res != VK_SUCCESS {
            (self.vk.destroy_buffer)(self.device, buffer, ptr::null());
            return Err(format!("vkAllocateMemory failed: {}", res));
        }

        let res = (self.vk.bind_buffer_memory)(self.device, buffer, memory, 0);
        if res != VK_SUCCESS {
            return Err(format!("vkBindBufferMemory failed: {}", res));
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
        (self.vk.begin_command_buffer)(cmd_buf, &begin_info);

        let region = VkBufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        (self.vk.cmd_copy_buffer)(cmd_buf, src.buffer, dst.buffer, 1, &region);
        (self.vk.end_command_buffer)(cmd_buf);

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
            return Err(format!("vkQueueSubmit (staging) failed: {}", res));
        }
        let res = (self.vk.queue_wait_idle)(self.queue);
        if res != VK_SUCCESS {
            return Err(format!("vkQueueWaitIdle (staging) failed: {}", res));
        }

        Ok(())
    }

    pub(crate) unsafe fn destroy_buffer(&self, buf: GpuBuffer) {
        (self.vk.free_memory)(self.device, buf.memory, ptr::null());
        (self.vk.destroy_buffer)(self.device, buf.buffer, ptr::null());
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
