#ifndef VULKAN_SYNC_H
#define VULKAN_SYNC_H

#include <vulkan/vulkan.h>
#include "VulkanError.h"

class ScopedGPUWait {
private:
    VkFence fence;
    VkDevice deviceRef;

public:
    ScopedGPUWait(VkDevice device);
    ~ScopedGPUWait();

    VkFence get() const;
    void wait() const;

    // Delete copy constructors
    ScopedGPUWait(const ScopedGPUWait&) = delete;
    ScopedGPUWait& operator=(const ScopedGPUWait&) = delete;
};

#endif // VULKAN_SYNC_H