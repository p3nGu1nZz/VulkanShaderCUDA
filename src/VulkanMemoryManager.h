#ifndef VULKAN_MEMORY_MANAGER_H
#define VULKAN_MEMORY_MANAGER_H

#include <vulkan/vulkan.h>
#include <vector>
#include "VulkanError.h"

class VulkanMemoryManager {
public:
    struct AllocationInfo {
        VkDeviceMemory memory;
        VkDeviceSize offset;
        VkDeviceSize size;
    };

    VulkanMemoryManager(VkPhysicalDevice physicalDevice, VkDevice device);
    ~VulkanMemoryManager();

    AllocationInfo allocate(VkDeviceSize size, VkMemoryPropertyFlags properties);

private:
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    std::vector<AllocationInfo> allocations;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

#endif // VULKAN_MEMORY_MANAGER_H