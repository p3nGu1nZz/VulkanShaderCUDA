// src\VulkanContext.cpp
#include "VulkanContext.h"
#include "vulkan_globals.h"
#include "spdlog/spdlog.h"

#include <stdexcept>
#include <vector>
#include <algorithm>

// Constructor
VulkanContext::VulkanContext()
    : instance(VK_NULL_HANDLE), physicalDevice(VK_NULL_HANDLE),
      device(VK_NULL_HANDLE), computeQueue(VK_NULL_HANDLE),
      computeQueueFamilyIndex(std::numeric_limits<uint32_t>::max()) {}

// Destructor
VulkanContext::~VulkanContext() {
    cleanupVulkan();
}

// Initialize Vulkan resources
void VulkanContext::initVulkan() {
    spdlog::info("Creating Vulkan instance...");

    // 1. Create Vulkan Instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanBackend";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "NoEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2; // Ensure compatibility

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Optional: Enable validation layers during development
    std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
#endif

    // Optional: Specify required extensions
    std::vector<const char*> extensions = {
        // Add necessary extensions here, e.g., "VK_KHR_surface", "VK_KHR_win32_surface" for Windows
    };
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }

    spdlog::info("Vulkan instance created.");

    // 2. Pick Physical Device
    pickPhysicalDevice();

    spdlog::info("Selected Vulkan physical device.");

    // 3. Find Compute Queue Family Index
    findComputeQueueFamily();

    spdlog::info("Found compute queue family index: {}", computeQueueFamilyIndex);

    // 4. Create Logical Device and Retrieve Queues
    createLogicalDevice();

    spdlog::info("Logical device created and compute queue retrieved.");

    // 5. Create Command Pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Allow resetting individual command buffers

    result = vkCreateCommandPool(device, &poolInfo, nullptr, &vulkan_globals::commandPool);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool.");
    }

    spdlog::info("Command pool created.");

    // 6. Create Descriptor Pool
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1000; // Adjust as needed based on application requirements

    VkDescriptorPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &poolSize;
    poolCreateInfo.maxSets = 1000; // Adjust as needed

    result = vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &vulkan_globals::descriptorPool);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool.");
    }

    spdlog::info("Descriptor pool created.");

    // 7. Create Descriptor Set Layout
    VkDescriptorSetLayoutBinding layoutBinding = {};
    layoutBinding.binding = 0;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &layoutBinding;

    VkDescriptorSetLayout descriptorSetLayout;
    result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout.");
    }

    spdlog::info("Descriptor set layout created.");

    // 8. Initialize Managers
    memoryManager = std::make_unique<VulkanMemoryManager>(physicalDevice, device);
    bufferPool = std::make_unique<VulkanBufferPool>(device);
    
    // Initialize ShaderManager
    std::shared_ptr<ShaderManager> shaderMgr = std::make_shared<ShaderManager>(device);
    
    // Initialize PipelineManager
    pipelineManager = std::make_unique<PipelineManager>(shaderMgr, device);
    
    // Initialize CommandBufferManager
    commandBufferManager = std::make_unique<CommandBufferManager>(device, vulkan_globals::commandPool);
    
    // Initialize DescriptorSetManager with the created layout
    descriptorSetManager = std::make_shared<DescriptorSetManager>(device, vulkan_globals::descriptorPool, descriptorSetLayout);

    spdlog::info("CommandBufferManager created.");
    spdlog::info("DescriptorSetManager created.");

    // Assign the Vulkan device and compute queue to the global variables
    vulkan_globals::device = device;
    vulkan_globals::computeQueue = computeQueue;

    spdlog::info("VulkanContext fully initialized.");
    spdlog::info("VulkanContext initialized successfully.");
}

// Cleanup Vulkan resources
void VulkanContext::cleanupVulkan() {
    if (device != VK_NULL_HANDLE) {
        spdlog::info("Cleaning up VulkanContext...");

        // Cleanup managers in reverse order of creation
        descriptorSetManager.reset();
        commandBufferManager.reset();
        pipelineManager.reset();
        bufferPool.reset();
        memoryManager.reset();

        // Destroy Descriptor Set Layout
        // Note: Assuming DescriptorSetManager does not store the layout; if it does, ensure it's destroyed there
        // If layout needs to be destroyed here, maintain a member variable for it

        vkDestroyDescriptorPool(device, vulkan_globals::descriptorPool, nullptr);
        vkDestroyCommandPool(device, vulkan_globals::commandPool, nullptr);

        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;

        spdlog::info("Logical device destroyed.");
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;

        spdlog::info("Vulkan instance destroyed.");
    }

    // Reset global device and compute queue variables
    vulkan_globals::device = VK_NULL_HANDLE;
    vulkan_globals::computeQueue = VK_NULL_HANDLE;

    spdlog::info("VulkanContext cleaned up.");
}

// Pick a suitable physical device
void VulkanContext::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Select the first suitable device
    for (const auto& dev : devices) {
        // Here, you can add more checks to select a better device based on criteria like Vulkan version, extensions, etc.
        physicalDevice = dev;
        break;
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to find a suitable GPU.");
    }
}

// Find a queue family that supports compute operations
void VulkanContext::findComputeQueueFamily() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    if (queueFamilyCount == 0) {
        throw std::runtime_error("Failed to find any queue families.");
    }

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    // Look for a queue family that supports compute
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            computeQueue = VK_NULL_HANDLE; // Will be retrieved after device creation
            return;
        }
    }

    throw std::runtime_error("Failed to find a compute queue family.");
}

// Create a logical device and retrieve compute queue
void VulkanContext::createLogicalDevice() {
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // Specify device features if needed
    VkPhysicalDeviceFeatures deviceFeatures = {};
    deviceFeatures.shaderStorageImageExtendedFormats = VK_TRUE; // Example feature

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;

    createInfo.pEnabledFeatures = &deviceFeatures;

    // Enable necessary device extensions
    std::vector<const char*> deviceExtensions = {
        // Add required device extensions here, e.g., "VK_KHR_swapchain" if needed
    };
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    // Optional: Enable validation layers (if not already enabled at instance level)
    std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
#ifndef NDEBUG
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
#else
    createInfo.enabledLayerCount = 0;
#endif

    VkResult result = vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device.");
    }

    // Retrieve the compute queue
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
    if (computeQueue == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to retrieve compute queue.");
    }

    spdlog::info("Compute queue retrieved successfully.");
}
