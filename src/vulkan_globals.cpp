#include "vulkan_globals.h"
#include "VulkanContext.h"
#include "spdlog/spdlog.h"
#include <filesystem>

namespace vulkan_globals {
    // Define global Vulkan variables
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

    // Initialize shader directory
    std::filesystem::path vulkan_globals::shader_directory;

    // Vulkan context instance
    std::unique_ptr<VulkanContext> vulkanContextInstance = nullptr;

    void setShaderDirectory(const std::filesystem::path& exe_path) {
        // Get the directory containing the executable
        auto base_path = exe_path.parent_path();
        
        // Try different relative paths to find shaders
        std::vector<std::filesystem::path> potential_paths = {
            base_path / "shaders",                    // Direct shaders subdirectory
            base_path / ".." / "shaders",             // One level up
            base_path / ".." / ".." / "shaders",      // Two levels up
            base_path / ".." / ".." / ".." / "shaders" // Three levels up
        };

        for (const auto& path : potential_paths) {
            if (std::filesystem::exists(path) && 
                std::filesystem::exists(path / "add.comp.spv")) {
                shader_directory = path;
                spdlog::info("Found shader directory at: {}", shader_directory.string());
                return;
            }
        }

        spdlog::error("Could not find valid shader directory in any of the searched locations");
        for (const auto& path : potential_paths) {
            spdlog::error("Searched: {}", path.string());
        }
        throw std::runtime_error("Could not find valid shader directory");
    }

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