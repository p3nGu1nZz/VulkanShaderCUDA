// VulkanDeviceHelper.h
#ifndef VULKAN_DEVICE_HELPER_H
#define VULKAN_DEVICE_HELPER_H

#include <vulkan/vulkan.h>
#include <vector>

class VulkanDeviceHelper {
public:
    static VkPhysicalDeviceProperties getPhysicalDeviceProperties(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        return properties;
    }

    static uint32_t findComputeQueueFamily(VkPhysicalDevice device) {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                return i;
            }
        }

        throw std::runtime_error("Could not find a compute queue family.");
    }
};

#endif // VULKAN_DEVICE_HELPER_H