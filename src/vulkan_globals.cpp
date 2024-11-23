// src\vulkan_globals.cpp
#include "vulkan_globals.h"
#include "VulkanContext.h"
#include "spdlog/spdlog.h"

namespace vulkan_globals {
    // Define global Vulkan variables
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    // Vulkan context instance
    std::unique_ptr<VulkanContext> vulkanContextInstance = nullptr;

    // Initialize Vulkan
    bool initializeVulkan() {
        try {
            if (!vulkanContextInstance) {
                spdlog::info("Creating VulkanContext instance...");
                vulkanContextInstance = std::make_unique<VulkanContext>();
                vulkanContextInstance->initVulkan();
            } else {
                spdlog::warn("VulkanContext is already initialized.");
            }
            return true;
        }
        catch (const std::exception& e) {
            spdlog::error("Failed to initialize Vulkan: {}", e.what());
            return false;
        }
    }

    // Cleanup Vulkan
    void cleanupVulkan() {
        if (vulkanContextInstance) {
            spdlog::info("Cleaning up VulkanContext...");
            vulkanContextInstance->cleanupVulkan();
            vulkanContextInstance.reset();
            spdlog::info("VulkanContext cleanup complete.");
        } else {
            spdlog::warn("VulkanContext is not initialized or already cleaned up.");
        }
    }

    // Get VulkanContext instance
    VulkanContext* getContext() {
        return vulkanContextInstance.get();
    }
}
