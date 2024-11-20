// vulkan_backend.cpp

#include <vulkan/vulkan.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <onnx.pb.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <filesystem>
#include <cassert>
#include <cstring>
// Include spdlog for logging
#include <spdlog/spdlog.h>

namespace py = pybind11;
namespace fs = std::filesystem;

// Macro for Vulkan error checking with detailed messages
#define VK_CHECK_DETAILED(result, operation) \
    if (result != VK_SUCCESS) { \
        throw std::runtime_error(std::string("Vulkan Error in ") + operation + \
            " (line " + std::to_string(__LINE__) + "): " + std::to_string(result) + " - " + getVulkanErrorString(result)); \
    }

// Function to convert VkResult to string for better error messages
std::string getVulkanErrorString(VkResult result) {
    switch(result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        // Add more cases as needed
        default: return "Unknown VkResult";
    }
}

// Utility function to check and align buffer sizes
VkDeviceSize checkSize(size_t size) {
    if (size > static_cast<size_t>(std::numeric_limits<VkDeviceSize>::max())) {
        throw std::runtime_error("Size exceeds VkDeviceSize limit");
    }
    return static_cast<VkDeviceSize>(size);
}

// Forward declarations
class VulkanTensor;
class VulkanBufferPool;

// Push Constants Structures
struct MatMulPushConstants {
    uint32_t M;
    uint32_t K;
    uint32_t N;
};

struct Conv2DPushConstants {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_depth;
    uint32_t output_channels;
    uint32_t kernel_size;
    uint32_t padding;
    uint32_t stride;
};

struct SoftmaxPushConstants {
    uint32_t size;
};

struct MaxPoolPushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t batch_size;
    uint32_t poolSizeX;
    uint32_t poolSizeY;
    uint32_t strideX;
    uint32_t strideY;
};

// RAII Wrapper for Vulkan Resources
class VulkanResource {
protected:
    VkDevice device;
    std::function<void()> deleter;

public:
    VulkanResource(VkDevice device, std::function<void()> deleter)
        : device(device), deleter(deleter) {}

    ~VulkanResource() {
        if (deleter) {
            deleter();
        }
    }

    // Delete copy semantics
    VulkanResource(const VulkanResource&) = delete;
    VulkanResource& operator=(const VulkanResource&) = delete;

    // Enable move semantics
    VulkanResource(VulkanResource&& other) noexcept
        : device(other.device), deleter(std::move(other.deleter)) {
        other.deleter = nullptr;
    }

    VulkanResource& operator=(VulkanResource&& other) noexcept {
        if (this != &other) {
            if (deleter) {
                deleter();
            }
            device = other.device;
            deleter = std::move(other.deleter);
            other.deleter = nullptr;
        }
        return *this;
    }
};

// Shader Module Cache for reusing shader modules
class ShaderModuleCache {
private:
    VkDevice device;
    std::unordered_map<std::string, VkShaderModule> cache;
    std::mutex mutex;

    // Helper function to load shader code from a file
    std::vector<char> loadShaderCode(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open shader file: " + filename);
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

public:
    ShaderModuleCache(VkDevice device) : device(device) {}

    ~ShaderModuleCache() {
        for (auto& [name, module] : cache) {
            vkDestroyShaderModule(device, module, nullptr);
        }
    }

    VkShaderModule getShaderModule(const std::string& shaderName) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(shaderName);
        if (it != cache.end()) {
            return it->second;
        }

        // Load shader code
        std::vector<char> shaderCode = loadShaderCode(shaderName);

        // Create shader module
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = shaderCode.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

        VkShaderModule shaderModule;
        VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
        VK_CHECK_DETAILED(result, "Shader Module Creation");

        cache[shaderName] = shaderModule;
        return shaderModule;
    }
};

// VulkanBufferPool for managing buffer allocations and reuse
class VulkanBufferPool {
private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> buffers;
    std::unordered_map<VkBuffer, bool> bufferUsage;  // Track if buffer is in use
    std::mutex mutex;

    // Helper function to find suitable memory type
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && 
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }

public:
    VulkanBufferPool(VkDevice device, VkPhysicalDevice physicalDevice)
        : device(device), physicalDevice(physicalDevice) {}

    ~VulkanBufferPool() {
        for (auto& [buffer, memory] : buffers) {
            vkDestroyBuffer(device, buffer, nullptr);
            vkFreeMemory(device, memory, nullptr);
        }
    }

    // Allocate a buffer and memory with buffer usage tracking
    std::pair<VkBuffer, VkDeviceMemory> allocateBuffer(
        VkDeviceSize size, 
        VkBufferUsageFlags usage, 
        VkMemoryPropertyFlags properties) {
        std::lock_guard<std::mutex> lock(mutex);

        // First try to find an existing unused buffer of suitable size
        for (size_t i = 0; i < buffers.size(); i++) {
            if (!bufferUsage[buffers[i].first]) {
                VkMemoryRequirements memReq;
                vkGetBufferMemoryRequirements(device, buffers[i].first, &memReq);
                if (memReq.size >= size) {
                    bufferUsage[buffers[i].first] = true;
                    return buffers[i];
                }
            }
        }

        // If no suitable buffer found, create new one
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer;
        VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
        VK_CHECK_DETAILED(result, "Buffer Creation");

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        VkDeviceMemory bufferMemory;
        result = vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
        VK_CHECK_DETAILED(result, "Memory Allocation");

        vkBindBufferMemory(device, buffer, bufferMemory, 0);

        buffers.emplace_back(buffer, bufferMemory);
        bufferUsage[buffer] = true;
        return {buffer, bufferMemory};
    }

    // Release a buffer back to the pool
    void releaseBuffer(VkBuffer buffer) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = bufferUsage.find(buffer);
        if (it != bufferUsage.end()) {
            it->second = false;  // Mark as unused
        }
    }
};

// Shader Push Constants
struct ShaderPushConstants {
    // This can be expanded based on shader requirements
};

// VulkanContext to manage Vulkan initialization and resources
struct VulkanContext {
    VkInstance instance;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;
    std::unique_ptr<ShaderModuleCache> shaderCache;
    std::unique_ptr<VulkanBufferPool> bufferPool;
    VkPipelineCache pipelineCache;

    // Initialize Vulkan
    void initVulkan() {
        // Create Vulkan instance
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan PyTorch Backend";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        // Enable extensions
        std::vector<const char*> extensions;
        #ifndef NDEBUG
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        #endif

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        VK_CHECK_DETAILED(result, "Instance Creation");

        // Select physical device
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // Select the first suitable device
        for (const auto& device : devices) {
            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(device, &deviceProperties);

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

            // Find a queue family that supports compute operations
            for (uint32_t i = 0; i < queueFamilyCount; i++) {
                if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physicalDevice = device;
                    computeQueueFamilyIndex = i;
                    break;
                }
            }

            if (physicalDevice != VK_NULL_HANDLE) {
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to find a suitable GPU with compute capabilities.");
        }

        // Create logical device and compute queue
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        // Specify device features if needed
        VkPhysicalDeviceFeatures deviceFeatures = {};

        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

        // Enable necessary device extensions
        std::vector<const char*> deviceExtensions = {
            VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
            VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME
        };
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

        result = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
        VK_CHECK_DETAILED(result, "Logical Device Creation");

        vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);

        // Create command pool
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        result = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
        VK_CHECK_DETAILED(result, "Command Pool Creation");

        // Create shader module cache
        shaderCache = std::make_unique<ShaderModuleCache>(device);

        // Create buffer pool
        bufferPool = std::make_unique<VulkanBufferPool>(device, physicalDevice);

        // Create pipeline cache
        VkPipelineCacheCreateInfo pipelineCacheInfo = {};
        pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        pipelineCacheInfo.initialDataSize = 0;
        pipelineCacheInfo.pInitialData = nullptr;
        pipelineCacheInfo.flags = 0;

        result = vkCreatePipelineCache(device, &pipelineCacheInfo, nullptr, &pipelineCache);
        VK_CHECK_DETAILED(result, "Pipeline Cache Creation");

        spdlog::info("Vulkan initialized successfully.");
    }

    // Cleanup Vulkan resources
    void cleanupVulkan() {
        vkDestroyPipelineCache(device, pipelineCache, nullptr);
        bufferPool.reset();
        shaderCache.reset();
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        spdlog::info("Vulkan resources cleaned up.");
    }
};

// Global VulkanContext instance
std::unique_ptr<VulkanContext> vulkanContext;

// Initialize Vulkan (to be called from Python)
void initVulkanBackend() {
    if (vulkanContext) {
        spdlog::warn("Vulkan backend is already initialized.");
        return;
    }
    vulkanContext = std::make_unique<VulkanContext>();
    vulkanContext->initVulkan();
}

// VulkanTensor class definition
class VulkanTensor {
private:
    VkDeviceSize size;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t depth;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDevice device;
    VulkanBufferPool* bufferPoolPtr;

public:
    // Constructor
    VulkanTensor(VkDeviceSize size, uint32_t w = 1, uint32_t h = 1, uint32_t c = 1, uint32_t d = 1, const void* data = nullptr)
        : size(size), width(w), height(h), channels(c), depth(d), 
          device(vulkanContext->device), bufferPoolPtr(vulkanContext->bufferPool.get()) {
        
        auto [buf, mem] = bufferPoolPtr->allocateBuffer(
            size, 
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        buffer = buf;
        memory = mem;

        if (data) {
            upload(data);
        }
    }

    // Move constructor
    VulkanTensor(VulkanTensor&& other) noexcept
        : size(other.size), width(other.width), height(other.height), 
          channels(other.channels), depth(other.depth),
          buffer(other.buffer), memory(other.memory), 
          device(other.device), bufferPoolPtr(other.bufferPoolPtr) {
        other.buffer = VK_NULL_HANDLE;
        other.memory = VK_NULL_HANDLE;
    }

    // Move assignment operator
    VulkanTensor& operator=(VulkanTensor&& other) noexcept {
        if (this != &other) {
            if (buffer != VK_NULL_HANDLE && memory != VK_NULL_HANDLE) {
                bufferPoolPtr->releaseBuffer(buffer);
            }

            size = other.size;
            width = other.width;
            height = other.height;
            channels = other.channels;
            depth = other.depth;
            buffer = other.buffer;
            memory = other.memory;
            device = other.device;
            bufferPoolPtr = other.bufferPoolPtr;

            other.buffer = VK_NULL_HANDLE;
            other.memory = VK_NULL_HANDLE;
        }
        return *this;
    }

    // Destructor
    ~VulkanTensor() {
        if (buffer != VK_NULL_HANDLE && memory != VK_NULL_HANDLE) {
            bufferPoolPtr->releaseBuffer(buffer);
        }
    }

    // Delete copy constructor and assignment
    VulkanTensor(const VulkanTensor&) = delete;
    VulkanTensor& operator=(const VulkanTensor&) = delete;

    // Data transfer methods
    void upload(const void* data) {
        if (!data) {
            throw std::runtime_error("Null pointer provided for upload");
        }

        void* mappedMemory;
        VkResult result = vkMapMemory(device, memory, 0, size, 0, &mappedMemory);
        VK_CHECK_DETAILED(result, "Memory Mapping for Upload");

        memcpy(mappedMemory, data, static_cast<size_t>(size));

        VkMappedMemoryRange memoryRange = {};
        memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        memoryRange.memory = memory;
        memoryRange.size = size;
        vkFlushMappedMemoryRanges(device, 1, &memoryRange);

        vkUnmapMemory(device, memory);
    }

    void download(void* data) const {
        if (!data) {
            throw std::runtime_error("Null pointer provided for download");
        }

        void* mappedMemory;
        VkResult result = vkMapMemory(device, memory, 0, size, 0, &mappedMemory);
        VK_CHECK_DETAILED(result, "Memory Mapping for Download");

        VkMappedMemoryRange memoryRange = {};
        memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        memoryRange.memory = memory;
        memoryRange.size = size;
        vkInvalidateMappedMemoryRanges(device, 1, &memoryRange);

        memcpy(data, mappedMemory, static_cast<size_t>(size));
        vkUnmapMemory(device, memory);
    }

    // Getters
    VkDeviceSize getSize() const { return size; }
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }
    uint32_t getChannels() const { return channels; }
    uint32_t getDepth() const { return depth; }
    VkBuffer getBuffer() const { return buffer; }

    // Utility methods
    bool isValid() const {
        return buffer != VK_NULL_HANDLE && memory != VK_NULL_HANDLE;
    }

    std::string getDimensionsString() const {
        return "(" + std::to_string(channels) + ", " + 
               std::to_string(height) + ", " + 
               std::to_string(width) + ", " +
               std::to_string(depth) + ")";
    }

    bool verifyDimensions(VkDeviceSize expectedSize) const {
        return size == expectedSize &&
               (width * height * channels * depth * sizeof(float)) == size;
    }
};

// Template function for executing shaders
template <typename PushConstants>
void executeShader(const std::string& operation,
                  const VulkanTensor& inputA,
                  const VulkanTensor& inputB,
                  VulkanTensor& output,
                  VkDeviceSize workgroupSizeX = 256,
                  VkDeviceSize workgroupSizeY = 1,
                  VkDeviceSize workgroupSizeZ = 1,
                  bool useThirdBuffer = true,
                  const PushConstants* pushConstants = nullptr) {
    spdlog::debug("Executing shader for operation: {}", operation);

    // Load shader module
    std::string shaderPath = "shaders/" + operation + ".spv";
    VkShaderModule shaderModule = vulkanContext->shaderCache->getShaderModule(shaderPath);

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1; // Assuming one descriptor set

    // Create a simple descriptor set layout with three storage buffers
    VkDescriptorSetLayoutBinding bindings[3] = {};

    for (int i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout;
    VkResult result = vkCreateDescriptorSetLayout(vulkanContext->device, &layoutInfo, nullptr, &descriptorSetLayout);
    VK_CHECK_DETAILED(result, "Descriptor Set Layout Creation");

    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPipelineLayout pipelineLayout;
    result = vkCreatePipelineLayout(vulkanContext->device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
    VK_CHECK_DETAILED(result, "Pipeline Layout Creation");

    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName  = "main";
    pipelineInfo.layout = pipelineLayout;

    VkPipeline computePipeline;
    result = vkCreateComputePipelines(vulkanContext->device, vulkanContext->pipelineCache, 1, &pipelineInfo, nullptr, &computePipeline);
    VK_CHECK_DETAILED(result, "Compute Pipeline Creation");

    // Allocate descriptor set
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    VkDescriptorPool descriptorPool;
    result = vkCreateDescriptorPool(vulkanContext->device, &poolInfo, nullptr, &descriptorPool);
    VK_CHECK_DETAILED(result, "Descriptor Pool Creation");

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    result = vkAllocateDescriptorSets(vulkanContext->device, &allocInfo, &descriptorSet);
    VK_CHECK_DETAILED(result, "Descriptor Set Allocation");

    // Update descriptor set
    VkDescriptorBufferInfo bufferInfos[3] = {};
    bufferInfos[0].buffer = inputA.getBuffer();
    bufferInfos[0].offset = 0;
    bufferInfos[0].range = inputA.getSize();

    bufferInfos[1].buffer = inputB.getBuffer();
    bufferInfos[1].offset = 0;
    bufferInfos[1].range = inputB.getSize();

    bufferInfos[2].buffer = output.getBuffer();
    bufferInfos[2].offset = 0;
    bufferInfos[2].range = output.getSize();

    VkWriteDescriptorSet descriptorWrites[3] = {};

    for (int i = 0; i < 3; ++i) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }

    vkUpdateDescriptorSets(vulkanContext->device, 3, descriptorWrites, 0, nullptr);

    // Allocate and begin command buffer
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = vulkanContext->commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    result = vkAllocateCommandBuffers(vulkanContext->device, &cmdAllocInfo, &commandBuffer);
    VK_CHECK_DETAILED(result, "Command Buffer Allocation");

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VK_CHECK_DETAILED(result, "Command Buffer Begin");

    // Bind pipeline and descriptor set
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Push constants
    if (pushConstants) {
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), pushConstants);
    }

    // Dispatch compute shader
    uint32_t dispatchX = static_cast<uint32_t>((inputA.getSize() + workgroupSizeX - 1) / workgroupSizeX);
    uint32_t dispatchY = static_cast<uint32_t>((inputA.getSize() + workgroupSizeY - 1) / workgroupSizeY);
    uint32_t dispatchZ = static_cast<uint32_t>(useThirdBuffer ? 1 : 1);

    vkCmdDispatch(commandBuffer, dispatchX, dispatchY, dispatchZ);

    // Memory barrier after computation
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    // End command buffer
    result = vkEndCommandBuffer(commandBuffer);
    VK_CHECK_DETAILED(result, "Command Buffer End");

    // Submit command buffer and wait for completion
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    // Create fence
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;

    VkFence fence;
    result = vkCreateFence(vulkanContext->device, &fenceInfo, nullptr, &fence);
    VK_CHECK_DETAILED(result, "Fence Creation");

    // Submit to compute queue
    result = vkQueueSubmit(vulkanContext->computeQueue, 1, &submitInfo, fence);
    VK_CHECK_DETAILED(result, "Queue Submission");

    // Wait for fence
    constexpr uint64_t FENCE_TIMEOUT = 5000000000; // 5 seconds in nanoseconds
    result = vkWaitForFences(vulkanContext->device, 1, &fence, VK_TRUE, FENCE_TIMEOUT);
    if (result == VK_TIMEOUT) {
        throw std::runtime_error("GPU operation timed out after 5 seconds");
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to wait for fence completion");
    }

    // Cleanup
    vkDestroyFence(vulkanContext->device, fence, nullptr);
    vkDestroyDescriptorPool(vulkanContext->device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(vulkanContext->device, descriptorSetLayout, nullptr);
    vkDestroyPipeline(vulkanContext->device, computePipeline, nullptr);
    vkDestroyPipelineLayout(vulkanContext->device, pipelineLayout, nullptr);

    spdlog::debug("Shader execution for operation '{}' completed.", operation);
}

// Specific operation execution functions
void executeAdd(const VulkanTensor& inputA, const VulkanTensor& inputB, VulkanTensor& output) {
    spdlog::info("Executing Add operation.");
    executeShader<ShaderPushConstants>(
        "add", 
        inputA, 
        inputB, 
        output, 
        256, 1, 1, 
        true, 
        nullptr
    );
}

void executeMatMul(const VulkanTensor& a, const VulkanTensor& b, VulkanTensor& c, uint32_t M, uint32_t K, uint32_t N) {
    spdlog::info("Executing MatMul operation.");
    MatMulPushConstants pushConstants = {M, K, N};
    executeShader<MatMulPushConstants>(
        "matmul", 
        a, 
        b, 
        c, 
        16, 16, 1, 
        true, 
        &pushConstants
    );
}

void executeReLU(const VulkanTensor& input, VulkanTensor& output) {
    spdlog::info("Executing ReLU operation.");
    executeShader<ShaderPushConstants>(
        "relu", 
        input, 
        VulkanTensor(0,1,1,1,1), 
        output, 
        256, 1, 1, 
        false, 
        nullptr
    );
}

void executeSigmoid(const VulkanTensor& input, VulkanTensor& output) {
    spdlog::info("Executing Sigmoid operation.");
    executeShader<ShaderPushConstants>(
        "sigmoid", 
        input, 
        VulkanTensor(0,1,1,1,1), 
        output, 
        256, 1, 1, 
        false, 
        nullptr
    );
}

void executeSoftmax(const VulkanTensor& input, VulkanTensor& output) {
    spdlog::info("Executing Softmax operation.");
    SoftmaxPushConstants pushConstants = {
        static_cast<uint32_t>(input.getSize() / sizeof(float))
    };
    executeShader<SoftmaxPushConstants>(
        "softmax", 
        input, 
        VulkanTensor(0,1,1,1,1), 
        output, 
        256, 1, 1, 
        false, 
        &pushConstants
    );
}

void executeConv2D(
    const VulkanTensor& input,
    const VulkanTensor& kernel,
    VulkanTensor& output,
    const Conv2DPushConstants& constants
) {
    spdlog::info("Executing Conv2D operation.");
    executeShader<Conv2DPushConstants>(
        "conv2d",
        input,
        kernel,
        output,
        8, 8, 1,
        true,
        &constants
    );
}

void executeMaxPool(const VulkanTensor& input, VulkanTensor& output, 
                   uint32_t width, uint32_t height, uint32_t channels,
                   uint32_t poolSizeX, uint32_t poolSizeY,
                   uint32_t strideX, uint32_t strideY) {
    spdlog::info("Executing MaxPool operation.");
    MaxPoolPushConstants pushConstants = {
        width,
        height,
        channels,
        1,          // batch_size
        poolSizeX,
        poolSizeY,
        strideX,
        strideY
    };
    executeShader<MaxPoolPushConstants>(
        "pooling",
        input,
        VulkanTensor(0,1,1,1,1),
        output,
        16, 16, channels,
        false,
        &pushConstants
    );
}

void executeBatchNorm(const VulkanTensor& input, const VulkanTensor& gamma, const VulkanTensor& beta, VulkanTensor& output,
                     uint32_t size, float epsilon) {
    spdlog::info("Executing BatchNorm operation.");
    struct BatchNormPushConstants {
        uint32_t size;
        float epsilon;
    } pushConstants = {size, epsilon};

    executeShader<BatchNormPushConstants>(
        "batchnorm",
        input,
        gamma,
        output,
        256, 1, 1,
        true,
        &pushConstants
    );
}

// ONNX Model Parser Class
class OnnxModelParser {
private:
    onnx::ModelProto modelProto;

public:
    OnnxModelParser(const std::string& modelPath) {
        // Load ONNX model from file
        std::ifstream modelFile(modelPath, std::ios::binary);
        if (!modelFile.is_open()) {
            throw std::runtime_error("Failed to open ONNX model file: " + modelPath);
        }

        if (!modelProto.ParseFromIstream(&modelFile)) {
            throw std::runtime_error("Failed to parse ONNX model.");
        }

        modelFile.close();
    }

    const onnx::GraphProto& getGraph() const {
        return modelProto.graph();
    }

    std::vector<onnx::NodeProto> getNodes() const {
        const auto& graph = getGraph();
        return {graph.node().begin(), graph.node().end()};
    }

    onnx::TensorProto getInitializer(const std::string& name) const {
        const auto& graph = getGraph();
        for (const auto& initializer : graph.initializer()) {
            if (initializer.name() == name) {
                return initializer;
            }
        }
        throw std::runtime_error("Initializer not found: " + name);
    }
};

// Extract Node Attributes
struct NodeAttributes {
    std::unordered_map<std::string, int> intAttributes;
    std::unordered_map<std::string, float> floatAttributes;
    std::unordered_map<std::string, std::vector<int>> intArrayAttributes;
};

NodeAttributes extractAttributes(const onnx::NodeProto& node) {
    NodeAttributes attributes;
    for (const auto& attr : node.attribute()) {
        if (attr.type() == onnx::AttributeProto::INT) {
            attributes.intAttributes[attr.name()] = attr.i();
        } else if (attr.type() == onnx::AttributeProto::FLOAT) {
            attributes.floatAttributes[attr.name()] = attr.f();
        } else if (attr.type() == onnx::AttributeProto::INTS) {
            attributes.intArrayAttributes[attr.name()] = {attr.ints().begin(), attr.ints().end()};
        }
        // Handle other attribute types if necessary
    }
    return attributes;
}

// Convert ONNX Nodes to Vulkan Operations
void convertMatMul(const onnx::NodeProto& node, VulkanTensor& inputA, VulkanTensor& inputB, VulkanTensor& output) {
    uint32_t M = inputA.getHeight();
    uint32_t K = inputA.getWidth();
    uint32_t N = inputB.getWidth();

    executeMatMul(inputA, inputB, output, M, K, N);
}

void convertConv2D(const onnx::NodeProto& node, VulkanTensor& input, VulkanTensor& kernel, VulkanTensor& output) {
    auto attributes = extractAttributes(node);

    uint32_t kernelSize = 3; // Default value
    if (attributes.intArrayAttributes.find("kernel_shape") != attributes.intArrayAttributes.end()) {
        const auto& ks = attributes.intArrayAttributes.at("kernel_shape");
        if (ks.size() >= 2) {
            kernelSize = ks[1]; // Assuming square kernels
        }
    }

    uint32_t padding = 1; // Default value
    if (attributes.intArrayAttributes.find("pads") != attributes.intArrayAttributes.end()) {
        const auto& pads = attributes.intArrayAttributes.at("pads");
        if (pads.size() >= 2) {
            padding = pads[0]; // Assuming same padding for height and width
        }
    }

    uint32_t stride = 1; // Default value
    if (attributes.intArrayAttributes.find("strides") != attributes.intArrayAttributes.end()) {
        const auto& strides = attributes.intArrayAttributes.at("strides");
        if (strides.size() >= 2) {
            stride = strides[0]; // Assuming same stride for height and width
        }
    }

    Conv2DPushConstants pushConstants = {
        input.getWidth(),
        input.getHeight(),
        input.getChannels(),
        kernel.getWidth(), // Assuming weights.getWidth() represents output_channels
        kernelSize,
        padding,
        stride
    };

    executeConv2D(input, kernel, output, pushConstants);
}

void convertReLU(const onnx::NodeProto& node, VulkanTensor& input, VulkanTensor& output) {
    executeReLU(input, output);
}

void convertSigmoid(const onnx::NodeProto& node, VulkanTensor& input, VulkanTensor& output) {
    executeSigmoid(input, output);
}

void convertSoftmax(const onnx::NodeProto& node, VulkanTensor& input, VulkanTensor& output) {
    executeSoftmax(input, output);
}

void convertPooling(const onnx::NodeProto& node, VulkanTensor& input, VulkanTensor& output) {
    auto attributes = extractAttributes(node);

    uint32_t poolSizeX = 2;
    uint32_t poolSizeY = 2;
    uint32_t strideX = 2;
    uint32_t strideY = 2;

    if (attributes.intArrayAttributes.find("kernel_shape") != attributes.intArrayAttributes.end()) {
        const auto& ks = attributes.intArrayAttributes.at("kernel_shape");
        if (ks.size() >= 2) {
            poolSizeX = ks[1];
            poolSizeY = ks[0];
        }
    }

    if (attributes.intArrayAttributes.find("strides") != attributes.intArrayAttributes.end()) {
        const auto& strides = attributes.intArrayAttributes.at("strides");
        if (strides.size() >= 2) {
            strideX = strides[1];
            strideY = strides[0];
        }
    }

    uint32_t width = input.getWidth();
    uint32_t height = input.getHeight();
    uint32_t channels = input.getChannels();

    executeMaxPool(input, output, width, height, channels, poolSizeX, poolSizeY, strideX, strideY);
}

void convertBatchNorm(const onnx::NodeProto& node, VulkanTensor& input, VulkanTensor& gamma, VulkanTensor& beta, VulkanTensor& output) {
    auto attributes = extractAttributes(node);
    float epsilon = 1e-5f;
    if (attributes.floatAttributes.find("epsilon") != attributes.floatAttributes.end()) {
        epsilon = attributes.floatAttributes.at("epsilon");
    }

    uint32_t size = static_cast<uint32_t>(input.getSize() / sizeof(float));
    executeBatchNorm(input, gamma, beta, output, size, epsilon);
}

// Tensor Management Utilities
VulkanTensor createTensor(const onnx::TensorProto& tensorProto) {
    VkDeviceSize size = checkSize(tensorProto.raw_data().size());
    // For simplicity, assume 1D tensors. Extend this based on actual tensor dimensions.
    VulkanTensor tensor(size, 1, 1, 1, 1);

    // Upload raw data to the tensor
    tensor.upload(tensorProto.raw_data().data());

    return tensor;
}

VulkanTensor allocateOutputTensor(const onnx::NodeProto& node, const std::string& outputName) {
    // Placeholder: Infer output dimensions from node attributes or input tensors
    // For simplicity, assume output dimensions are same as input
    // In practice, you should extract dimensions based on the operation

    // Example dimensions (these should be properly inferred)
    uint32_t width = 16;
    uint32_t height = 16;
    uint32_t channels = 16;
    VulkanTensor outputTensor(checkSize(width * height * channels * sizeof(float)), width, height, channels, 1);
    return outputTensor;
}

// Parse and Execute ONNX Graph
void parseAndExecuteGraph(const onnx::GraphProto& graph) {
    // Use unique_ptr for automatic cleanup
    std::unordered_map<std::string, std::unique_ptr<VulkanTensor>> tensors;

    // Load initializers (weights/biases)
    for (const auto& initializer : graph.initializer()) {
        tensors[initializer.name()] = std::make_unique<VulkanTensor>(createTensor(initializer));
    }

    // Process each node
    for (const auto& node : graph.node()) {
        if (node.op_type() == "MatMul") {
            VulkanTensor& inputA = *tensors.at(node.input(0));
            VulkanTensor& inputB = *tensors.at(node.input(1));
            VulkanTensor output = allocateOutputTensor(node, node.output(0));

            convertMatMul(node, inputA, inputB, output);
            tensors[node.output(0)] = std::make_unique<VulkanTensor>(std::move(output));
        }
        else if (node.op_type() == "Conv") {
            VulkanTensor& input = *tensors.at(node.input(0));
            VulkanTensor& kernel = *tensors.at(node.input(1));
            VulkanTensor output = allocateOutputTensor(node, node.output(0));

            convertConv2D(node, input, kernel, output);
            tensors[node.output(0)] = std::make_unique<VulkanTensor>(std::move(output));
        }
        else if (node.op_type() == "Relu") {
            VulkanTensor& input = *tensors.at(node.input(0));
            VulkanTensor output = allocateOutputTensor(node, node.output(0));

            convertReLU(node, input, output);
            tensors[node.output(0)] = std::make_unique<VulkanTensor>(std::move(output));
        }
        else if (node.op_type() == "Sigmoid") {
            VulkanTensor& input = *tensors.at(node.input(0));
            VulkanTensor output = allocateOutputTensor(node, node.output(0));

            convertSigmoid(node, input, output);
            tensors[node.output(0)] = std::make_unique<VulkanTensor>(std::move(output));
        }
        else if (node.op_type() == "Softmax") {
            VulkanTensor& input = *tensors.at(node.input(0));
            VulkanTensor output = allocateOutputTensor(node, node.output(0));

            convertSoftmax(node, input, output);
            tensors[node.output(0)] = std::make_unique<VulkanTensor>(std::move(output));
        }
        else if (node.op_type() == "MaxPool") {
            VulkanTensor& input = *tensors.at(node.input(0));
            VulkanTensor output = allocateOutputTensor(node, node.output(0));

            convertPooling(node, input, output);
            tensors[node.output(0)] = std::make_unique<VulkanTensor>(std::move(output));
        }
        else if (node.op_type() == "BatchNormalization") {
            VulkanTensor& input = *tensors.at(node.input(0));
            VulkanTensor& gamma = *tensors.at(node.input(1));
            VulkanTensor& beta = *tensors.at(node.input(2));
            VulkanTensor output = allocateOutputTensor(node, node.output(0));

            convertBatchNorm(node, input, gamma, beta, output);
            tensors[node.output(0)] = std::make_unique<VulkanTensor>(std::move(output));
        }
        else {
            throw std::runtime_error("Unsupported ONNX operation: " + node.op_type());
        }
    }

    spdlog::info("ONNX model executed successfully!");
}

// Complete the import_onnx_model function
void import_onnx_model(const std::string& modelPath) {
    OnnxModelParser parser(modelPath);
    const auto& graph = parser.getGraph();

    parseAndExecuteGraph(graph);
}

// PyBind11 bindings
PYBIND11_MODULE(vulkan_backend, m) {
    m.doc() = "Enhanced Vulkan backend for PyTorch with improved error handling and memory management";

    // Initialize Vulkan
    m.def("init_vulkan", &initVulkanBackend, "Initialize Vulkan backend with enhanced error checking");

    // Import ONNX model
    m.def("import_onnx_model", &import_onnx_model, py::arg("model_path"), 
          "Import and execute an ONNX model on Vulkan backend");

    // VulkanTensor class binding
    py::class_<VulkanTensor>(m, "VulkanTensor")
        .def(py::init<VkDeviceSize, uint32_t, uint32_t, uint32_t, uint32_t, const void*>(),
             py::arg("size"),
             py::arg("width") = 1,
             py::arg("height") = 1,
             py::arg("channels") = 1,
             py::arg("depth") = 1,
             py::arg("data") = nullptr)
        .def("get_size", &VulkanTensor::getSize)
        .def("get_width", &VulkanTensor::getWidth)
        .def("get_height", &VulkanTensor::getHeight)
        .def("get_channels", &VulkanTensor::getChannels)
        .def("get_depth", &VulkanTensor::getDepth)
        .def("get_buffer", [](const VulkanTensor& self) -> py::capsule {
            // Wrap VkBuffer as a PyCapsule with the name "VkBuffer"
            return py::capsule(reinterpret_cast<void*>(self.getBuffer()), "VkBuffer");
        }, "Get Vulkan buffer handle as a PyCapsule")
        .def("is_valid", &VulkanTensor::isValid)
        .def("get_dimensions_string", &VulkanTensor::getDimensionsString)
        .def("verify_dimensions", &VulkanTensor::verifyDimensions)
        .def("upload", [](VulkanTensor &self, py::array_t<float> data) {
            py::buffer_info buf = data.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Data must be a 1D array");
            }
            VkDeviceSize expected_size = self.getSize();
            VkDeviceSize actual_size = checkSize(buf.size * sizeof(float));
            if (actual_size != expected_size) {
                throw std::runtime_error("Data size mismatch. Expected: " + 
                    std::to_string(expected_size) + ", Got: " + std::to_string(actual_size));
            }
            self.upload(buf.ptr);
        }, "Upload data to VulkanTensor")
        .def("download", [](VulkanTensor &self) {
            py::array_t<float> result(self.getSize() / sizeof(float));
            py::buffer_info buf = result.request();
            self.download(buf.ptr);
            return result;
        }, "Download data from VulkanTensor");

    // Addition operation
    m.def("vulkan_add", [](py::array_t<float> a, py::array_t<float> b, py::array_t<float> c) {
        try {
            auto buf_a = a.request();
            auto buf_b = b.request();
            auto buf_c = c.request();

            VkDeviceSize size_a = checkSize(buf_a.size * sizeof(float));
            VkDeviceSize size_b = checkSize(buf_b.size * sizeof(float));
            VkDeviceSize size_c = checkSize(buf_c.size * sizeof(float));

            if (size_a != size_b || size_a != size_c) {
                throw std::runtime_error(
                    "Size mismatch. Input A: " + 
                    std::to_string(size_a) +
                    ", Input B: " + std::to_string(size_b) +
                    ", Output: " + std::to_string(size_c)
                );
            }

            VulkanTensor tensorA(size_a, static_cast<uint32_t>(buf_a.size), 1, 1, 1);
            VulkanTensor tensorB(size_b, static_cast<uint32_t>(buf_b.size), 1, 1, 1);
            VulkanTensor tensorC(size_c, static_cast<uint32_t>(buf_c.size), 1, 1, 1);

            tensorA.upload(buf_a.ptr);
            tensorB.upload(buf_b.ptr);

            executeAdd(tensorA, tensorB, tensorC);

            tensorC.download(buf_c.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Addition operation failed: " + std::string(e.what()));
        }
    }, "Execute addition operation on Vulkan backend");

    // Matrix Multiplication operation
    m.def("vulkan_matmul", [](py::array_t<float> inputA, py::array_t<float> inputB, py::array_t<float> output,
                               uint32_t M, uint32_t K, uint32_t N) {
        try {
            auto buf_a = inputA.request();
            auto buf_b = inputB.request();
            auto buf_c = output.request();

            VkDeviceSize size_a = checkSize(buf_a.size * sizeof(float));
            VkDeviceSize size_b = checkSize(buf_b.size * sizeof(float));
            VkDeviceSize size_c = checkSize(buf_c.size * sizeof(float));

            if (size_a != static_cast<VkDeviceSize>(M * K * sizeof(float))) {
                throw std::runtime_error(
                    "Input A size mismatch. Expected: " + 
                    std::to_string(M * K * sizeof(float)) +
                    ", Got: " + std::to_string(size_a)
                );
            }

            if (size_b != static_cast<VkDeviceSize>(K * N * sizeof(float))) {
                throw std::runtime_error(
                    "Input B size mismatch. Expected: " + 
                    std::to_string(K * N * sizeof(float)) +
                    ", Got: " + std::to_string(size_b)
                );
            }

            if (size_c != static_cast<VkDeviceSize>(M * N * sizeof(float))) {
                throw std::runtime_error(
                    "Output size mismatch. Expected: " + 
                    std::to_string(M * N * sizeof(float)) +
                    ", Got: " + std::to_string(size_c)
                );
            }

            VulkanTensor tensorA(size_a, K, M, 1, 1);
            VulkanTensor tensorB(size_b, N, K, 1, 1);
            VulkanTensor tensorC(size_c, N, M, 1, 1);

            tensorA.upload(buf_a.ptr);
            tensorB.upload(buf_b.ptr);

            executeMatMul(tensorA, tensorB, tensorC, M, K, N);

            tensorC.download(buf_c.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Matrix multiplication failed: " + std::string(e.what()));
        }
    }, "Execute matrix multiplication on Vulkan backend");

    // ReLU operation
    m.def("vulkan_relu", [](py::array_t<float> input, py::array_t<float> output) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            VkDeviceSize size_input = checkSize(buf_input.size * sizeof(float));
            VkDeviceSize size_output = checkSize(buf_output.size * sizeof(float));

            if (size_input != size_output) {
                throw std::runtime_error(
                    "Size mismatch. Input: " + 
                    std::to_string(size_input) +
                    ", Output: " + std::to_string(size_output)
                );
            }

            VulkanTensor tensorInput(size_input, static_cast<uint32_t>(buf_input.size), 1, 1, 1);
            VulkanTensor tensorOutput(size_output, static_cast<uint32_t>(buf_output.size), 1, 1, 1);

            tensorInput.upload(buf_input.ptr);
            executeReLU(tensorInput, tensorOutput);
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("ReLU operation failed: " + std::string(e.what()));
        }
    }, "Execute ReLU activation on Vulkan backend");

    // Sigmoid operation
    m.def("vulkan_sigmoid", [](py::array_t<float> input, py::array_t<float> output) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            VkDeviceSize size_input = checkSize(buf_input.size * sizeof(float));
            VkDeviceSize size_output = checkSize(buf_output.size * sizeof(float));

            if (size_input != size_output) {
                throw std::runtime_error(
                    "Size mismatch. Input: " + 
                    std::to_string(size_input) +
                    ", Output: " + std::to_string(size_output)
                );
            }

            VulkanTensor tensorInput(size_input, static_cast<uint32_t>(buf_input.size), 1, 1, 1);
            VulkanTensor tensorOutput(size_output, static_cast<uint32_t>(buf_output.size), 1, 1, 1);

            tensorInput.upload(buf_input.ptr);
            executeSigmoid(tensorInput, tensorOutput);
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Sigmoid operation failed: " + std::string(e.what()));
        }
    }, "Execute Sigmoid activation on Vulkan backend");

    // Softmax operation
    m.def("vulkan_softmax", [](py::array_t<float> input, py::array_t<float> output) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            VkDeviceSize size_input = checkSize(buf_input.size * sizeof(float));
            VkDeviceSize size_output = checkSize(buf_output.size * sizeof(float));

            if (size_input != size_output) {
                throw std::runtime_error(
                    "Size mismatch. Input: " + 
                    std::to_string(size_input) +
                    ", Output: " + std::to_string(size_output)
                );
            }

            VulkanTensor tensorInput(size_input, static_cast<uint32_t>(buf_input.size), 1, 1, 1);
            VulkanTensor tensorOutput(size_output, static_cast<uint32_t>(buf_output.size), 1, 1, 1);

            tensorInput.upload(buf_input.ptr);
            executeSoftmax(tensorInput, tensorOutput);
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Softmax operation failed: " + std::string(e.what()));
        }
    }, "Execute Softmax operation on Vulkan backend");

    // Conv2D operation
    m.def("vulkan_conv2d", [](
        py::array_t<float> input,
        py::array_t<float> kernel,
        py::array_t<float> output,
        uint32_t input_width,
        uint32_t input_height, 
        uint32_t input_channels,
        uint32_t output_channels,
        uint32_t kernel_size,
        uint32_t padding,
        uint32_t stride) {
        try {
            // Get buffer info
            auto buf_input = input.request();
            auto buf_kernel = kernel.request();
            auto buf_output = output.request();
    
            // Check contiguous
            if (!(input.flags() & py::array::c_style)) {
                throw std::runtime_error("Input must be contiguous");
            }
            if (!(kernel.flags() & py::array::c_style)) {
                throw std::runtime_error("Kernel must be contiguous");
            }
            if (!(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Output must be contiguous");
            }
    
            // Calculate output dimensions
            uint32_t output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
            uint32_t output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    
            // Calculate expected sizes
            size_t expected_input_elements = static_cast<size_t>(input_width) * 
                                        static_cast<size_t>(input_height) * 
                                        static_cast<size_t>(input_channels);
            size_t expected_kernel_elements = static_cast<size_t>(kernel_size) * 
                                            static_cast<size_t>(kernel_size) * 
                                            static_cast<size_t>(input_channels) * 
                                            static_cast<size_t>(output_channels);
            size_t expected_output_elements = static_cast<size_t>(output_width) * 
                                            static_cast<size_t>(output_height) * 
                                            static_cast<size_t>(output_channels);
    
            if (buf_input.size != expected_input_elements) {
                throw std::runtime_error(
                    "Input size mismatch. Expected: " + std::to_string(expected_input_elements) +
                    ", Got: " + std::to_string(buf_input.size)
                );
            }
    
            if (buf_kernel.size != expected_kernel_elements) {
                throw std::runtime_error(
                    "Kernel size mismatch. Expected: " + std::to_string(expected_kernel_elements) +
                    ", Got: " + std::to_string(buf_kernel.size)
                );
            }
    
            if (buf_output.size != expected_output_elements) {
                throw std::runtime_error(
                    "Output size mismatch. Expected: " + std::to_string(expected_output_elements) +
                    ", Got: " + std::to_string(buf_output.size)
                );
            }
    
            // Create tensors with correct dimensions
            VulkanTensor tensorInput(
                checkSize(buf_input.size * sizeof(float)),
                input_width,
                input_height,
                input_channels,
                1  // Batch size = 1
            );
            VulkanTensor tensorKernel(
                checkSize(buf_kernel.size * sizeof(float)),
                kernel_size,
                kernel_size,
                input_channels,
                output_channels
            );
            VulkanTensor tensorOutput(
                checkSize(buf_output.size * sizeof(float)),
                output_width,
                output_height,
                output_channels,
                1  // Batch size = 1
            );
    
            // Upload data
            tensorInput.upload(buf_input.ptr);
            tensorKernel.upload(buf_kernel.ptr);
    
            // Set push constants
            Conv2DPushConstants pushConstants{
                input_width,
                input_height,
                input_channels,
                output_channels,
                kernel_size,
                padding,
                stride
            };
    
            // Execute convolution
            executeConv2D(tensorInput, tensorKernel, tensorOutput, pushConstants);
    
            // Download results
            tensorOutput.download(buf_output.ptr);
    
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Conv2D operation failed: ") + e.what());
        }
    }, 
        py::arg("input"), 
        py::arg("kernel"),
        py::arg("output"),
        py::arg("input_width"),
        py::arg("input_height"),
        py::arg("input_channels"),
        py::arg("output_channels"),
        py::arg("kernel_size"),
        py::arg("padding"),
        py::arg("stride"),
        R"pbdoc(
        Execute Conv2D operation on Vulkan backend.
        
        Args:
            input: Input tensor of shape (C, H, W)
            kernel: Kernel tensor of shape (K, C, R, S) 
            output: Output tensor of shape (K, P, Q)
            input_width: Width of input
            input_height: Height of input
            input_channels: Number of input channels
            output_channels: Number of output channels
            kernel_size: Size of kernel (assumes square)
            padding: Padding size
            stride: Convolution stride
        )pbdoc"
    );

    // Pooling operation
    m.def("vulkan_pooling", [](py::array_t<float> input, py::array_t<float> output,
                               uint32_t width, uint32_t height, uint32_t depth,
                               uint32_t poolSizeX, uint32_t poolSizeY,
                               uint32_t strideX, uint32_t strideY) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            VkDeviceSize size_input = checkSize(buf_input.size * sizeof(float));
            VkDeviceSize size_output = checkSize(buf_output.size * sizeof(float));

            uint32_t expected_output_width = (width - poolSizeX) / strideX + 1;
            uint32_t expected_output_height = (height - poolSizeY) / strideY + 1;
            uint32_t expected_output_size = expected_output_width * expected_output_height * depth;

            if (size_input != static_cast<VkDeviceSize>(width * height * depth * sizeof(float))) {
                throw std::runtime_error(
                    "Input dimensions mismatch. Expected size: " + 
                    std::to_string(width * height * depth * sizeof(float)) +
                    ", Got: " + std::to_string(size_input)
                );
            }

            if (size_output != static_cast<VkDeviceSize>(expected_output_size * sizeof(float))) {
                throw std::runtime_error(
                    "Output dimensions mismatch. Expected size: " + 
                    std::to_string(expected_output_size * sizeof(float)) +
                    ", Got: " + std::to_string(size_output)
                );
            }

            VulkanTensor tensorInput(size_input, width, height, depth, 1);
            VulkanTensor tensorOutput(size_output, expected_output_width, expected_output_height, depth, 1);

            // Upload data
            tensorInput.upload(buf_input.ptr);

            // Execute pooling
            executeMaxPool(tensorInput, tensorOutput, width, height, depth, poolSizeX, poolSizeY, strideX, strideY);

            // Download result
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Pooling operation failed: " + std::string(e.what()));
        }
    }, "Execute MaxPool2D operation on Vulkan backend");

    // BatchNorm operation
    m.def("vulkan_batchnorm", [](py::array_t<float> input, py::array_t<float> gamma, py::array_t<float> beta, py::array_t<float> output,
                                 uint32_t size, float epsilon) {
        try {
            auto buf_input = input.request();
            auto buf_gamma = gamma.request();
            auto buf_beta = beta.request();
            auto buf_output = output.request();

            VkDeviceSize size_input = checkSize(buf_input.size * sizeof(float));
            VkDeviceSize size_gamma = checkSize(buf_gamma.size * sizeof(float));
            VkDeviceSize size_beta = checkSize(buf_beta.size * sizeof(float));
            VkDeviceSize size_output = checkSize(buf_output.size * sizeof(float));

            if (size_input != static_cast<VkDeviceSize>(size * sizeof(float)) || 
                size_gamma != static_cast<VkDeviceSize>(size * sizeof(float)) ||
                size_beta != static_cast<VkDeviceSize>(size * sizeof(float)) || 
                size_output != static_cast<VkDeviceSize>(size * sizeof(float))) {
                throw std::runtime_error(
                    "Size mismatch. All input, gamma, beta, and output tensors must have the same size."
                );
            }

            // Ensure the arrays are contiguous
            if (!(input.flags() & py::array::c_style)) {
                throw std::runtime_error("Input array must be contiguous.");
            }
            if (!(gamma.flags() & py::array::c_style)) {
                throw std::runtime_error("Gamma array must be contiguous.");
            }
            if (!(beta.flags() & py::array::c_style)) {
                throw std::runtime_error("Beta array must be contiguous.");
            }
            if (!(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Output array must be contiguous.");
            }

            VulkanTensor tensorInput(size_input, size, 1, 1, 1);
            VulkanTensor tensorGamma(size_gamma, size, 1, 1, 1);
            VulkanTensor tensorBeta(size_beta, size, 1, 1, 1);
            VulkanTensor tensorOutput(size_output, size, 1, 1, 1);

            // Upload data
            tensorInput.upload(buf_input.ptr);
            tensorGamma.upload(buf_gamma.ptr);
            tensorBeta.upload(buf_beta.ptr);

            // Execute BatchNorm
            executeBatchNorm(tensorInput, tensorGamma, tensorBeta, tensorOutput, size, epsilon);

            // Download result
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("BatchNorm operation failed: " + std::string(e.what()));
        }
    }, "Execute BatchNorm operation on Vulkan backend");

    // Placeholder for model saving (not implemented)
    m.def("save_model", [](const std::string& filename) -> void {
        throw std::runtime_error("Model saving not yet implemented.");
    }, "Save the current model to a file");

    // Placeholder for model loading (not implemented)
    m.def("load_model", [](const std::string& filename) -> void {
        throw std::runtime_error("Model loading not yet implemented.");
    }, "Load a model from a file");

    // Placeholder for distributed training initialization (not implemented)
    m.def("initialize_distributed", [](uint32_t num_gpus) -> void {
        throw std::runtime_error("Distributed training not yet implemented.");
    }, "Initialize distributed training with the specified number of GPUs");

    // Placeholder for gradient checkpointing (not implemented)
    m.def("enable_gradient_checkpointing", []() -> void {
        throw std::runtime_error("Gradient checkpointing not yet implemented.");
    }, "Enable gradient checkpointing to optimize memory usage during training");

    // Placeholder for buffer pooling creation (not implemented)
    m.def("create_buffer_pool", [](size_t poolSize) -> py::capsule {
        throw std::runtime_error("Buffer pooling not yet implemented.");
    }, "Create a Vulkan buffer pool with the specified size");

    // Placeholder for buffer allocation from pool (not implemented)
    m.def("allocate_buffer", [](py::capsule pool, size_t size) -> py::capsule {
        throw std::runtime_error("Buffer allocation not yet implemented.");
    }, "Allocate a buffer from the specified buffer pool");

    // Placeholder for buffer deallocation to pool (not implemented)
    m.def("free_buffer", [](py::capsule pool, py::capsule buffer) -> void {
        throw std::runtime_error("Buffer deallocation not yet implemented.");
    }, "Free a buffer and return it to the specified buffer pool");
}
