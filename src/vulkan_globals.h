// src\vulkan_globals.h
#ifndef VULKAN_GLOBALS_H
#define VULKAN_GLOBALS_H

#include <vulkan/vulkan.h>
#include <memory>
#include "VulkanContext.h"

// Namespace for global Vulkan objects
namespace vulkan_globals {
    extern VkInstance instance;
    extern VkPhysicalDevice physicalDevice;
    extern VkDevice device;
    extern VkQueue computeQueue;
    extern VkQueue graphicsQueue;
    extern VkCommandPool commandPool;
    extern VkDescriptorPool descriptorPool;

    // Vulkan context instance
    extern std::unique_ptr<VulkanContext> vulkanContextInstance;

    // Functions to manage Vulkan initialization and cleanup
    bool initializeVulkan();
    void cleanupVulkan();
    VulkanContext* getContext();
}

#endif // VULKAN_GLOBALS_H
