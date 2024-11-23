#ifndef VULKAN_BUFFER_POOL_H
#define VULKAN_BUFFER_POOL_H

#include <vulkan/vulkan.h>
#include <queue>
#include "VulkanError.h"

class VulkanBufferPool {
public:
    VulkanBufferPool(VkDevice device);
    ~VulkanBufferPool();

    VkBuffer acquireBuffer();
    void releaseBuffer(VkBuffer buffer);

private:
    VkDevice device;
    std::queue<VkBuffer> buffers;
};

#endif // VULKAN_BUFFER_POOL_H