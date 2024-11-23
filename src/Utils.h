#ifndef UTILS_H
#define UTILS_H

#include <vulkan/vulkan.h>
#include <stdexcept>
#include <limits>
#include "VulkanError.h"

#define VK_CHECK_DETAILED(result, opType) \
    if ((result) != VK_SUCCESS) { \
        throw VulkanError("Vulkan operation failed with error code " + std::to_string(result) + \
                         " during operation " + std::to_string(static_cast<int>(opType))); \
    }

inline VkDeviceSize checkSize(size_t size) {
    if (size > static_cast<size_t>(std::numeric_limits<VkDeviceSize>::max())) {
        throw std::runtime_error("Size exceeds VkDeviceSize limit");
    }
    return static_cast<VkDeviceSize>(size);
}

#endif // UTILS_H