// src\VulkanMemoryManager.cpp
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
    // Create a temporary buffer to get memory requirements
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer tempBuffer;
    VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &tempBuffer);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to create temporary buffer for memory allocation.");
    }

    // Get memory requirements for the buffer
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, tempBuffer, &memRequirements);
    vkDestroyBuffer(device, tempBuffer, nullptr);

    // Find suitable memory type
    uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
        throw VulkanError("Failed to find suitable memory type.");
    }

    // Allocate memory with proper alignment
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;  // Use the size from requirements
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    AllocationInfo allocation = {};
    result = vkAllocateMemory(device, &allocInfo, nullptr, &allocation.memory);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to allocate device memory.");
    }

    allocation.size = memRequirements.size;
    allocation.offset = 0;

    // Store allocation for cleanup
    allocations.push_back(allocation);

    return allocation;
}

uint32_t VulkanMemoryManager::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    // First try to find an exactly matching memory type
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    // If exact match not found, try to find a memory type with additional properties
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties)) {
            return i;
        }
    }

    return std::numeric_limits<uint32_t>::max();
}
