#ifndef VULKAN_SYNC_H
#define VULKAN_SYNC_H

#include <vulkan/vulkan.h>
#include "VulkanError.h"

namespace VulkanSync {

class ScopedGPUWait {
public:
    ScopedGPUWait(VkDevice device);
    ~ScopedGPUWait();

    void wait() const;
    VkFence get() const;

    // Delete copy operations
    ScopedGPUWait(const ScopedGPUWait&) = delete;
    ScopedGPUWait& operator=(const ScopedGPUWait&) = delete;

private:
    VkDevice deviceRef;
    VkFence fence;
};

namespace MemoryBarrier {
    VkMemoryBarrier getBarrier(VkAccessFlags srcAccess, VkAccessFlags dstAccess);
    VkBufferMemoryBarrier getBufferBarrier(
        VkBuffer buffer,
        VkAccessFlags srcAccess,
        VkAccessFlags dstAccess,
        VkDeviceSize offset,
        VkDeviceSize size
    );
    void cmdPipelineBarrier(
        VkCommandBuffer cmdBuffer,
        VkPipelineStageFlags srcStage,
        VkPipelineStageFlags dstStage,
        const VkBufferMemoryBarrier& bufferBarrier
    );
}

} // namespace VulkanSync

#endif // VULKAN_SYNC_H