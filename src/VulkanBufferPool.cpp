// src\VulkanBufferPool.cpp

#include "VulkanBufferPool.h"

// Constructor
VulkanBufferPool::VulkanBufferPool(VkDevice device)
    : device(device) {}

// Destructor: Cleans up all buffers in the pool
VulkanBufferPool::~VulkanBufferPool() {
    while (!buffers.empty()) {
        VkBuffer buffer = buffers.front();
        buffers.pop();
        vkDestroyBuffer(device, buffer, nullptr);
    }
}

// Acquire a buffer from the pool or create a new one if the pool is empty
VkBuffer VulkanBufferPool::acquireBuffer() {
    if (!buffers.empty()) {
        VkBuffer buf = buffers.front();
        buffers.pop();
        return buf;
    }

    // Create a new buffer with predefined size and usage
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = 1024 * 1024; // 1MB buffer, adjust as needed
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to create buffer in VulkanBufferPool.");
    }

    return buffer;
}

// Release a buffer back to the pool for reuse
void VulkanBufferPool::releaseBuffer(VkBuffer buffer) {
    buffers.push(buffer);
}
