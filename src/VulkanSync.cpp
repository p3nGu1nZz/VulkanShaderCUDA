#include "VulkanSync.h"

ScopedGPUWait::ScopedGPUWait(VkDevice device) : deviceRef(device), fence(VK_NULL_HANDLE) {
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;

    VkResult result = vkCreateFence(deviceRef, &fenceInfo, nullptr, &fence);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to create fence during ScopedGPUWait.");
    }
}

ScopedGPUWait::~ScopedGPUWait() {
    if (fence != VK_NULL_HANDLE) {
        vkDestroyFence(deviceRef, fence, nullptr);
        fence = VK_NULL_HANDLE;
    }
}

void ScopedGPUWait::wait() const {
    if (fence == VK_NULL_HANDLE) {
        throw VulkanError("Attempting to wait on null fence.");
    }

    VkResult result = vkWaitForFences(deviceRef, 1, &fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to wait for fence in ScopedGPUWait.");
    }

    result = vkResetFences(deviceRef, 1, &fence);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to reset fence in ScopedGPUWait.");
    }
}

VkFence ScopedGPUWait::get() const {
    return fence;
}