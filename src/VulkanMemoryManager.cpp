#include "VulkanMemoryManager.h"

VulkanMemoryManager::VulkanMemoryManager(VkPhysicalDevice physicalDevice, VkDevice device)
    : physicalDevice(physicalDevice), device(device) {}

VulkanMemoryManager::~VulkanMemoryManager() {
    for (auto& allocation : allocations) {
        if (allocation.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, allocation.memory, nullptr);
        }
    }
}

VulkanMemoryManager::AllocationInfo VulkanMemoryManager::allocate(VkDeviceSize size, VkMemoryPropertyFlags properties) {
    VkBuffer dummyBuffer;
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &dummyBuffer);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to create dummy buffer for memory allocation.");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, dummyBuffer, &memRequirements);
    vkDestroyBuffer(device, dummyBuffer, nullptr);

    uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
        throw VulkanError("Failed to find suitable memory type.");
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    AllocationInfo allocation = {};
    result = vkAllocateMemory(device, &allocInfo, nullptr, &allocation.memory);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to allocate device memory.");
    }

    allocation.size = memRequirements.size;
    allocation.offset = 0;
    allocations.push_back(allocation);

    return allocation;
}

uint32_t VulkanMemoryManager::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return std::numeric_limits<uint32_t>::max();
}