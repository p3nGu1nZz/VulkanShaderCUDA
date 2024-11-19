#include <vulkan/vulkan.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <limits> // For std::numeric_limits
#include <type_traits> // For std::is_same_v

namespace py = pybind11;

// Enhanced error checking macro with detailed reporting
#define VK_CHECK_DETAILED(result, operation) \
    if (result != VK_SUCCESS) { \
        throw std::runtime_error( \
            std::string("Vulkan Error in ") + operation + \
            " (line " + std::to_string(__LINE__) + "): " + \
            std::to_string(result)); \
    }

// Shader path helper function
std::string getShaderPath(const std::string& shaderName) {
    std::vector<std::string> searchPaths = {
        "shaders/",
        "../shaders/",
        "../../shaders/",
        "./shaders/",
        "./",
        ""
    };

    for (const auto& basePath : searchPaths) {
        std::string fullPath = basePath + shaderName;
        std::ifstream f(fullPath.c_str());
        if (f.good()) {
            return fullPath;
        }
    }
    return shaderName;
}

// Push constant structures
struct MatMulPushConstants {
    uint32_t M;  // Height of A
    uint32_t K;  // Width of A / Height of B
    uint32_t N;  // Width of B
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

struct Conv2DPushConstants {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel_size;
    uint32_t batch_size;
    uint32_t padding;
    uint32_t stride;
};

// RAII wrapper for Vulkan resources
class VulkanResource {
private:
    VkDevice device;
    std::function<void()> cleanup;
    bool moved;

public:
    VulkanResource(VkDevice d, std::function<void()> c) 
        : device(d), cleanup(c), moved(false) {}
    
    ~VulkanResource() {
        if (!moved && cleanup) {
            cleanup();
        }
    }

    VulkanResource(VulkanResource&& other) noexcept
        : device(other.device), cleanup(std::move(other.cleanup)), moved(false) {
        other.moved = true;
    }

    VulkanResource(const VulkanResource&) = delete;
    VulkanResource& operator=(const VulkanResource&) = delete;
};

// Buffer pool for reusing allocated buffers
class VulkanBufferPool {
private:
    struct BufferInfo {
        VkBuffer buffer;
        VkDeviceMemory memory;
        uint32_t size;
        bool in_use;
    };

    VkDevice device;
    VkPhysicalDevice physicalDevice;
    std::vector<BufferInfo> buffers;
    std::mutex mutex;

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
    VulkanBufferPool(VkDevice d, VkPhysicalDevice pd) 
        : device(d), physicalDevice(pd) {}

    ~VulkanBufferPool() {
        for (auto& info : buffers) {
            vkDestroyBuffer(device, info.buffer, nullptr);
            vkFreeMemory(device, info.memory, nullptr);
        }
    }

    std::pair<VkBuffer, VkDeviceMemory> acquireBuffer(uint32_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        
        for (auto& info : buffers) {
            if (!info.in_use && info.size >= size) {
                info.in_use = true;
                return {info.buffer, info.memory};
            }
        }

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = static_cast<VkDeviceSize>(size);
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer;
        VK_CHECK_DETAILED(
            vkCreateBuffer(device, &bufferInfo, nullptr, &buffer),
            "Buffer Creation"
        );

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(
            memRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        VkDeviceMemory memory;
        VK_CHECK_DETAILED(
            vkAllocateMemory(device, &allocInfo, nullptr, &memory),
            "Memory Allocation"
        );

        vkBindBufferMemory(device, buffer, memory, 0);

        buffers.push_back({buffer, memory, size, true});
        return {buffer, memory};
    }

    void releaseBuffer(VkBuffer buffer) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& info : buffers) {
            if (info.buffer == buffer) {
                info.in_use = false;
                break;
            }
        }
    }
};

// Vulkan context
struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::unique_ptr<VulkanBufferPool> bufferPool;

    ~VulkanContext() {
        if (commandPool != VK_NULL_HANDLE)
            vkDestroyCommandPool(device, commandPool, nullptr);
        if (device != VK_NULL_HANDLE)
            vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE)
            vkDestroyInstance(instance, nullptr);
    }
};

std::unique_ptr<VulkanContext> vulkanContext;

// Tensor class
class VulkanTensor {
private:
    uint32_t size;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDevice device;
    VulkanBufferPool* bufferPoolPtr;

public:
    VulkanTensor(uint32_t size, uint32_t w = 1, uint32_t h = 1, uint32_t d = 1)
        : size(size), width(w), height(h), depth(d) {
        if (!vulkanContext || !vulkanContext->device) {
            throw std::runtime_error("Vulkan not initialized");
        }
        device = vulkanContext->device;
        bufferPoolPtr = vulkanContext->bufferPool.get();
        
        auto bufferPair = bufferPoolPtr->acquireBuffer(size);
        buffer = bufferPair.first;
        memory = bufferPair.second;
    }

    ~VulkanTensor() {
        if (buffer != VK_NULL_HANDLE) {
            bufferPoolPtr->releaseBuffer(buffer);
        }
    }

    VulkanTensor(VulkanTensor&& other) noexcept
        : size(other.size), width(other.width), height(other.height), depth(other.depth),
          buffer(other.buffer), memory(other.memory), device(other.device),
          bufferPoolPtr(other.bufferPoolPtr) {
        other.buffer = VK_NULL_HANDLE;
        other.memory = VK_NULL_HANDLE;
    }

    VulkanTensor(const VulkanTensor&) = delete;
    VulkanTensor& operator=(const VulkanTensor&) = delete;

    void upload(const void* data) {
        void* mappedMemory;
        VK_CHECK_DETAILED(
            vkMapMemory(device, memory, 0, size, 0, &mappedMemory),
            "Memory Mapping for Upload"
        );
        memcpy(mappedMemory, data, static_cast<size_t>(size));
        vkUnmapMemory(device, memory);
    }

    void download(void* data) const {
        void* mappedMemory;
        VK_CHECK_DETAILED(
            vkMapMemory(device, memory, 0, size, 0, &mappedMemory),
            "Memory Mapping for Download"
        );
        memcpy(data, mappedMemory, static_cast<size_t>(size));
        vkUnmapMemory(device, memory);
    }

    uint32_t getSize() const { return size; }
    VkBuffer getBuffer() const { return buffer; }
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }
    uint32_t getDepth() const { return depth; }
};

// Shader mapping
std::unordered_map<std::string, std::string> shaderMapping = {
    {"add", "add.spv"},
    {"mul", "mul.spv"},
    {"matmul", "matmul.spv"},
    {"relu", "relu.spv"},
    {"sigmoid", "sigmoid.spv"},
    {"softmax", "softmax.spv"},
    {"conv2d", "conv2d.spv"},
    {"pooling", "pooling.spv"}
};

// Initialize Vulkan
void initVulkan() {
    if (vulkanContext && vulkanContext->instance != VK_NULL_HANDLE) {
        std::cout << "Vulkan already initialized." << std::endl;
        return;
    }

    vulkanContext = std::make_unique<VulkanContext>();

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Enhanced Vulkan Backend";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;

    VK_CHECK_DETAILED(
        vkCreateInstance(&instanceInfo, nullptr, &vulkanContext->instance),
        "Instance Creation"
    );

    uint32_t deviceCount = 0;
    VK_CHECK_DETAILED(
        vkEnumeratePhysicalDevices(vulkanContext->instance, &deviceCount, nullptr),
        "Physical Device Enumeration"
    );
    
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan-capable GPUs found");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    VK_CHECK_DETAILED(
        vkEnumeratePhysicalDevices(vulkanContext->instance, &deviceCount, devices.data()),
        "Physical Device Selection"
    );
    vulkanContext->physicalDevice = devices[0];

    // Retrieve queue family properties to find a compute queue
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vulkanContext->physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(vulkanContext->physicalDevice, &queueFamilyCount, queueFamilies.data());

    int computeQueueFamily = -1;
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamily = static_cast<int>(i);
            break;
        }
    }

    if (computeQueueFamily == -1) {
        throw std::runtime_error("No compute queue family found");
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = static_cast<uint32_t>(computeQueueFamily);
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueCreateInfo;

    VK_CHECK_DETAILED(
        vkCreateDevice(vulkanContext->physicalDevice, &deviceInfo, nullptr, &vulkanContext->device),
        "Logical Device Creation"
    );

    vkGetDeviceQueue(vulkanContext->device, static_cast<uint32_t>(computeQueueFamily), 0, &vulkanContext->computeQueue);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = static_cast<uint32_t>(computeQueueFamily);
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CHECK_DETAILED(
        vkCreateCommandPool(vulkanContext->device, &poolInfo, nullptr, &vulkanContext->commandPool),
        "Command Pool Creation"
    );

    vulkanContext->bufferPool = std::make_unique<VulkanBufferPool>(
        vulkanContext->device, 
        vulkanContext->physicalDevice
    );

    std::cout << "Enhanced Vulkan backend initialized successfully!" << std::endl;
}

// Shader loading
VkShaderModule loadShaderModule(const std::string& filename) {
    std::string shaderPath = getShaderPath(filename);
    std::ifstream file(shaderPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + shaderPath);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize % 4 != 0) {
        throw std::runtime_error("Shader code size is not a multiple of 4");
    }

    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    VkShaderModule shaderModule;
    VK_CHECK_DETAILED(
        vkCreateShaderModule(vulkanContext->device, &createInfo, nullptr, &shaderModule),
        "Shader Module Creation"
    );

    return shaderModule;
}

// Templated executeShader to handle different PushConstant structures
template <typename PushConstants>
void executeShader(const std::string& operation,
                   const VulkanTensor& inputA,
                   const VulkanTensor& inputB,
                   VulkanTensor& output,
                   uint32_t workgroupSizeX = 256,
                   uint32_t workgroupSizeY = 1,
                   uint32_t workgroupSizeZ = 1,
                   bool useThirdBuffer = true,
                   const PushConstants* pushConstants = nullptr) {

    // Map operation to shader filename
    auto it = shaderMapping.find(operation);
    if (it == shaderMapping.end()) {
        throw std::runtime_error("Unknown operation: " + operation);
    }

    std::string shaderFilename = it->second;
    std::string shaderPath = getShaderPath(shaderFilename);

    // Load shader module
    VkShaderModule shaderModule = loadShaderModule(shaderFilename);
    std::unique_ptr<VulkanResource> shaderResource(
        new VulkanResource(
            vulkanContext->device,
            [device = vulkanContext->device, shaderModule]() {
                vkDestroyShaderModule(device, shaderModule, nullptr);
            }
        )
    );

    // Create descriptor set layout
    VkDescriptorSetLayout descriptorSetLayout;
    {
        int bindingCount = useThirdBuffer ? 3 : 2;
        std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);

        for (int i = 0; i < bindingCount; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo = {};
        descriptorLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorLayoutInfo.bindingCount = bindingCount;
        descriptorLayoutInfo.pBindings = bindings.data();

        VK_CHECK_DETAILED(
            vkCreateDescriptorSetLayout(vulkanContext->device, &descriptorLayoutInfo, nullptr, &descriptorSetLayout),
            "Descriptor Set Layout Creation"
        );
    }

    // Create pipeline layout with push constants if needed
    VkPipelineLayout pipelineLayout;
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if constexpr (!std::is_same_v<PushConstants, void>) {
            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(PushConstants);
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        }
        else {
            pipelineLayoutInfo.pushConstantRangeCount = 0;
            pipelineLayoutInfo.pPushConstantRanges = nullptr;
        }

        VK_CHECK_DETAILED(
            vkCreatePipelineLayout(vulkanContext->device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
            "Pipeline Layout Creation"
        );
    }

    // Create compute pipeline
    VkPipeline pipeline;
    {
        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;

        VK_CHECK_DETAILED(
            vkCreateComputePipelines(vulkanContext->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
            "Compute Pipeline Creation"
        );
    }

    // Descriptor pool and set
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    {
        int bindingCount = useThirdBuffer ? 3 : 2;

        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = bindingCount;

        VkDescriptorPoolCreateInfo poolCreateInfo = {};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCreateInfo.poolSizeCount = 1;
        poolCreateInfo.pPoolSizes = &poolSize;
        poolCreateInfo.maxSets = 1;

        VK_CHECK_DETAILED(
            vkCreateDescriptorPool(vulkanContext->device, &poolCreateInfo, nullptr, &descriptorPool),
            "Descriptor Pool Creation"
        );

        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        VK_CHECK_DETAILED(
            vkAllocateDescriptorSets(vulkanContext->device, &allocInfo, &descriptorSet),
            "Descriptor Set Allocation"
        );

        // Update descriptor sets
        std::vector<VkDescriptorBufferInfo> bufferInfos(useThirdBuffer ? 3 : 2);
        bufferInfos[0] = {inputA.getBuffer(), 0, inputA.getSize()};
        if (useThirdBuffer) {
            bufferInfos[1] = {inputB.getBuffer(), 0, inputB.getSize()};
            bufferInfos[2] = {output.getBuffer(), 0, output.getSize()};
        } else {
            bufferInfos[1] = {output.getBuffer(), 0, output.getSize()};
        }

        std::vector<VkWriteDescriptorSet> descriptorWrites(useThirdBuffer ? 3 : 2);
        for (int i = 0; i < (useThirdBuffer ? 3 : 2); ++i) {
            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = descriptorSet;
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].dstArrayElement = 0;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        }

        vkUpdateDescriptorSets(vulkanContext->device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }

    // Record command buffer
    VkCommandBuffer commandBuffer;
    {
        VkCommandBufferAllocateInfo commandAllocInfo = {};
        commandAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandAllocInfo.commandPool = vulkanContext->commandPool;
        commandAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandAllocInfo.commandBufferCount = 1;

        VK_CHECK_DETAILED(
            vkAllocateCommandBuffers(vulkanContext->device, &commandAllocInfo, &commandBuffer),
            "Command Buffer Allocation"
        );

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CHECK_DETAILED(
            vkBeginCommandBuffer(commandBuffer, &beginInfo),
            "Command Buffer Recording Begin"
        );

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 
                               0, 1, &descriptorSet, 0, nullptr);

        // Handle push constants if applicable
        if constexpr (!std::is_same_v<PushConstants, void>) {
            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(PushConstants), pushConstants);
        }

        // Determine dispatch dimensions based on PushConstants type
        uint32_t dispatchX = workgroupSizeX;
        uint32_t dispatchY = workgroupSizeY;
        uint32_t dispatchZ = workgroupSizeZ;

        if constexpr (std::is_same_v<PushConstants, MatMulPushConstants>) {
            dispatchX = (pushConstants->M + 15) / 16;
            dispatchY = (pushConstants->N + 15) / 16;
            dispatchZ = 1;
        }
        else if constexpr (std::is_same_v<PushConstants, Conv2DPushConstants>) {
            dispatchX = 16;
            dispatchY = 16;
            dispatchZ = 4;
        }
        else if constexpr (std::is_same_v<PushConstants, SoftmaxPushConstants>) {
            dispatchX = (pushConstants->size + 255) / 256;
            dispatchY = 1;
            dispatchZ = 1;
        }
        else if constexpr (std::is_same_v<PushConstants, MaxPoolPushConstants>) {
            dispatchX = (pushConstants->poolSizeX + 15) / 16;
            dispatchY = (pushConstants->poolSizeY + 15) / 16;
            dispatchZ = pushConstants->channels;
        }

        vkCmdDispatch(commandBuffer, dispatchX, dispatchY, dispatchZ);

        VK_CHECK_DETAILED(
            vkEndCommandBuffer(commandBuffer),
            "Command Buffer Recording End"
        );
    }

    // Submit and wait
    {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VK_CHECK_DETAILED(
            vkQueueSubmit(vulkanContext->computeQueue, 1, &submitInfo, VK_NULL_HANDLE),
            "Command Buffer Submission"
        );

        VK_CHECK_DETAILED(
            vkQueueWaitIdle(vulkanContext->computeQueue),
            "Queue Wait Idle"
        );
    }

    // Cleanup
    vkFreeCommandBuffers(vulkanContext->device, vulkanContext->commandPool, 1, &commandBuffer);
    vkDestroyPipeline(vulkanContext->device, pipeline, nullptr);
    vkDestroyPipelineLayout(vulkanContext->device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(vulkanContext->device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(vulkanContext->device, descriptorPool, nullptr);
}

// Execute specific operation
void executeOperation(const std::string& operation, const VulkanTensor& inputA, const VulkanTensor& inputB, VulkanTensor& output) {
    if (operation == "matmul") {
        // Example: Set M, K, N based on application logic
        // These should be set appropriately before calling executeShader<MatMulPushConstants>
        // For demonstration, using dummy values
        MatMulPushConstants pushConstants = {16, 16, 16};
        executeShader<MatMulPushConstants>(
            operation, 
            inputA, 
            inputB, 
            output, 
            16, 16, 1, 
            true, 
            &pushConstants
        );
    }
    else if (operation == "conv2d") {
        // Example push constants for Conv2D
        // These should be set based on actual convolution parameters
        Conv2DPushConstants pushConstants = {
            inputA.getWidth(),
            inputA.getHeight(),
            inputA.getDepth(),
            inputB.getWidth(), // Assuming weights.getWidth() represents output_channels
            inputB.getHeight(), // Assuming weights.getHeight() represents kernel_size
            1, // batch_size
            0, // padding (example value)
            1  // stride (example value)
        };
        executeShader<Conv2DPushConstants>(
            operation, 
            inputA, 
            inputB, 
            output, 
            16, 16, 4, 
            true, 
            &pushConstants
        );
    }
    else if (operation == "softmax") {
        // Example push constants for Softmax
        SoftmaxPushConstants pushConstants = {
            inputA.getSize() / sizeof(float)
        };
        // For softmax, inputB is unused, so we pass a dummy VulkanTensor
        VulkanTensor dummyTensor(0);
        executeShader<SoftmaxPushConstants>(
            operation, 
            inputA, 
            dummyTensor, 
            output, 
            256, 1, 1, 
            false, 
            &pushConstants
        );
    }
    else if (operation == "pooling") {
        // Example push constants for MaxPool
        MaxPoolPushConstants pushConstants = {
            inputA.getWidth(),
            inputA.getHeight(),
            inputA.getDepth(),
            1,  // batch_size
            2,  // poolSizeX (example value)
            2,  // poolSizeY (example value)
            2,  // strideX (example value)
            2   // strideY (example value)
        };
        // For pooling, inputB is unused, so we pass a dummy VulkanTensor
        VulkanTensor dummyTensor(0);
        executeShader<MaxPoolPushConstants>(
            operation, 
            inputA, 
            dummyTensor, 
            output, 
            16, 16, pushConstants.channels, 
            false, 
            &pushConstants
        );
    }
    else {
        // Default case for operations like add, mul, relu, sigmoid
        // For these operations, no push constants are needed
        executeShader<void>(
            operation, 
            inputA, 
            inputB, 
            output, 
            16, 16, 1, 
            (operation == "add" || operation == "mul"), 
            nullptr
        );
    }
}

// Functional interfaces
void executeMatMul(const VulkanTensor& a, const VulkanTensor& b, VulkanTensor& c, uint32_t M, uint32_t K, uint32_t N) {
    MatMulPushConstants constants = {M, K, N};
    executeShader<MatMulPushConstants>(
        "matmul", 
        a, 
        b, 
        c, 
        16, 16, 1, 
        true, 
        &constants
    );
}

void executeAdd(const VulkanTensor& a, const VulkanTensor& b, VulkanTensor& c) {
    executeOperation("add", a, b, c);
}

void executeReLU(const VulkanTensor& input, VulkanTensor& output) {
    executeOperation("relu", input, VulkanTensor(0), output);
}

void executeSigmoid(const VulkanTensor& input, VulkanTensor& output) {
    executeOperation("sigmoid", input, VulkanTensor(0), output);
}

void executeSoftmax(const VulkanTensor& input, VulkanTensor& output) {
    executeOperation("softmax", input, VulkanTensor(0), output);
}

void executeConv2D(const VulkanTensor& input, const VulkanTensor& weights, VulkanTensor& output) {
    executeOperation("conv2d", input, weights, output);
}

void executeMaxPool(const VulkanTensor& input, VulkanTensor& output,
                   uint32_t width, uint32_t height, uint32_t channels,
                   uint32_t poolSizeX, uint32_t poolSizeY,
                   uint32_t strideX, uint32_t strideY) {
    MaxPoolPushConstants constants = {
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
        VulkanTensor(0),
        output,
        16, 16, channels,
        false,
        &constants
    );
}

// PyBind11 module definition
PYBIND11_MODULE(vulkan_backend, m) {
    m.doc() = "Enhanced Vulkan backend for PyTorch with improved error handling and memory management";

    // Add helper function for size checking
    auto checkSize = [](py::ssize_t size) -> uint32_t {
        if (size < 0 || size > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Array size out of valid range for uint32_t");
        }
        return static_cast<uint32_t>(size);
    };

    m.def("init_vulkan", &initVulkan, "Initialize Vulkan backend with enhanced error checking");

    // VulkanTensor class binding
    py::class_<VulkanTensor>(m, "VulkanTensor")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("size"),
             py::arg("width") = 1,
             py::arg("height") = 1,
             py::arg("depth") = 1)
        .def("get_size", &VulkanTensor::getSize)
        .def("get_width", &VulkanTensor::getWidth)
        .def("get_height", &VulkanTensor::getHeight)
        .def("get_depth", &VulkanTensor::getDepth)
        .def("upload", [](VulkanTensor &self, py::array_t<float> data) {
            py::buffer_info buf = data.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Data must be a 1D array");
            }
            uint32_t expected_size = self.getSize();
            uint32_t actual_size = static_cast<uint32_t>(buf.size * sizeof(float));
            if (static_cast<size_t>(actual_size) != buf.size * sizeof(float)) { // Prevent size_t to uint32_t warning
                throw std::runtime_error("Data size exceeds uint32_t limits");
            }
            if (actual_size != expected_size) {
                throw std::runtime_error("Data size mismatch. Expected: " + 
                    std::to_string(expected_size) + ", Got: " + std::to_string(actual_size));
            }
            self.upload(buf.ptr);
        })
        .def("download", [](VulkanTensor &self) {
            py::array_t<float> result(self.getSize() / sizeof(float));
            py::buffer_info buf = result.request();
            self.download(buf.ptr);
            return result;
        });

    // Addition operation
    m.def("vulkan_add", [checkSize](py::array_t<float> a, py::array_t<float> b, py::array_t<float> c) {
        try {
            auto buf_a = a.request();
            auto buf_b = b.request();
            auto buf_c = c.request();

            uint32_t size_a = checkSize(buf_a.size);
            uint32_t size_b = checkSize(buf_b.size);
            uint32_t size_c = checkSize(buf_c.size);

            if (size_a != size_b || size_a != size_c) {
                throw std::runtime_error(
                    "Size mismatch. Input A: " + std::to_string(size_a) +
                    ", Input B: " + std::to_string(size_b) +
                    ", Output: " + std::to_string(size_c)
                );
            }

            VulkanTensor tensorA(size_a * sizeof(float), size_a, 1, 1);
            VulkanTensor tensorB(size_b * sizeof(float), size_b, 1, 1);
            VulkanTensor tensorC(size_c * sizeof(float), size_c, 1, 1);

            tensorA.upload(buf_a.ptr);
            tensorB.upload(buf_b.ptr);

            executeAdd(tensorA, tensorB, tensorC);

            tensorC.download(buf_c.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Addition operation failed: " + std::string(e.what()));
        }
    });

    // Matrix Multiplication operation
    m.def("vulkan_matmul", [checkSize](py::array_t<float> inputA, py::array_t<float> inputB, py::array_t<float> output,
                                       uint32_t M, uint32_t K, uint32_t N) {
        try {
            auto buf_a = inputA.request();
            auto buf_b = inputB.request();
            auto buf_c = output.request();

            uint32_t size_a = checkSize(buf_a.size);
            uint32_t size_b = checkSize(buf_b.size);
            uint32_t size_c = checkSize(buf_c.size);

            if (size_a != M * K) {
                throw std::runtime_error(
                    "Input A size mismatch. Expected: " + 
                    std::to_string(M * K) +
                    ", Got: " + std::to_string(size_a)
                );
            }

            if (size_b != K * N) {
                throw std::runtime_error(
                    "Input B size mismatch. Expected: " + 
                    std::to_string(K * N) +
                    ", Got: " + std::to_string(size_b)
                );
            }

            if (size_c != M * N) {
                throw std::runtime_error(
                    "Output size mismatch. Expected: " + 
                    std::to_string(M * N) +
                    ", Got: " + std::to_string(size_c)
                );
            }

            VulkanTensor tensorA(size_a * sizeof(float), K, M, 1);
            VulkanTensor tensorB(size_b * sizeof(float), N, K, 1);
            VulkanTensor tensorC(size_c * sizeof(float), N, M, 1);

            tensorA.upload(buf_a.ptr);
            tensorB.upload(buf_b.ptr);

            executeMatMul(tensorA, tensorB, tensorC, M, K, N);

            tensorC.download(buf_c.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Matrix multiplication failed: " + std::string(e.what()));
        }
    });

    // ReLU operation
    m.def("vulkan_relu", [checkSize](py::array_t<float> input, py::array_t<float> output) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            uint32_t input_size = checkSize(buf_input.size);
            uint32_t output_size = checkSize(buf_output.size);

            if (input_size != output_size) {
                throw std::runtime_error(
                    "Size mismatch. Input: " + std::to_string(input_size) +
                    ", Output: " + std::to_string(output_size)
                );
            }

            VulkanTensor tensorInput(input_size * sizeof(float), input_size, 1, 1);
            VulkanTensor tensorOutput(output_size * sizeof(float), input_size, 1, 1);

            tensorInput.upload(buf_input.ptr);
            executeReLU(tensorInput, tensorOutput);
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("ReLU operation failed: " + std::string(e.what()));
        }
    });

    // Sigmoid operation
    m.def("vulkan_sigmoid", [checkSize](py::array_t<float> input, py::array_t<float> output) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            uint32_t input_size = checkSize(buf_input.size);
            uint32_t output_size = checkSize(buf_output.size);

            if (input_size != output_size) {
                throw std::runtime_error(
                    "Size mismatch. Input: " + std::to_string(input_size) +
                    ", Output: " + std::to_string(output_size)
                );
            }

            VulkanTensor tensorInput(input_size * sizeof(float), input_size, 1, 1);
            VulkanTensor tensorOutput(output_size * sizeof(float), input_size, 1, 1);

            tensorInput.upload(buf_input.ptr);
            executeSigmoid(tensorInput, tensorOutput);
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Sigmoid operation failed: " + std::string(e.what()));
        }
    });

    // Softmax operation
    m.def("vulkan_softmax", [checkSize](py::array_t<float> input, py::array_t<float> output) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            uint32_t input_size = checkSize(buf_input.size);
            uint32_t output_size = checkSize(buf_output.size);

            if (input_size != output_size) {
                throw std::runtime_error(
                    "Size mismatch. Input: " + std::to_string(input_size) +
                    ", Output: " + std::to_string(output_size)
                );
            }

            VulkanTensor tensorInput(input_size * sizeof(float), input_size, 1, 1);
            VulkanTensor tensorOutput(output_size * sizeof(float), input_size, 1, 1);

            tensorInput.upload(buf_input.ptr);
            executeSoftmax(tensorInput, tensorOutput);
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Softmax operation failed: " + std::string(e.what()));
        }
    });

    // Conv2D operation
    m.def("vulkan_conv2d", [checkSize](py::array_t<float> input, py::array_t<float> kernel, py::array_t<float> output,
                                      uint32_t inputWidth, uint32_t inputHeight, uint32_t inputDepth,
                                      uint32_t kernelWidth, uint32_t kernelHeight, uint32_t outputDepth,
                                      uint32_t padding, uint32_t stride) {
        try {
            auto buf_input = input.request();
            auto buf_kernel = kernel.request();
            auto buf_output = output.request();

            uint32_t size_input = checkSize(buf_input.size);
            uint32_t size_kernel = checkSize(buf_kernel.size);
            uint32_t size_output = checkSize(buf_output.size);

            // Calculate correct output dimensions
            int outputWidth = (static_cast<int>(inputWidth) - static_cast<int>(kernelWidth) + 2 * static_cast<int>(padding)) / static_cast<int>(stride) + 1;
            int outputHeight = (static_cast<int>(inputHeight) - static_cast<int>(kernelHeight) + 2 * static_cast<int>(padding)) / static_cast<int>(stride) + 1;

            if (outputWidth <= 0 || outputHeight <= 0) {
                throw std::runtime_error("Invalid output dimensions calculated for Conv2D");
            }

            if (size_output != static_cast<uint32_t>(outputWidth * outputHeight * outputDepth)) {
                throw std::runtime_error(
                    "Output dimensions do not match Conv2D requirements. "
                    "Expected: " + std::to_string(outputWidth * outputHeight * outputDepth) +
                    ", Got: " + std::to_string(size_output)
                );
            }

            VulkanTensor tensorInput(size_input * sizeof(float), inputWidth, inputHeight, inputDepth);
            VulkanTensor tensorKernel(size_kernel * sizeof(float), kernelWidth, kernelHeight, inputDepth);
            VulkanTensor tensorOutput(size_output * sizeof(float), outputWidth, outputHeight, outputDepth);

            tensorInput.upload(buf_input.ptr);
            tensorKernel.upload(buf_kernel.ptr);

            executeConv2D(tensorInput, tensorKernel, tensorOutput);

            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Conv2D operation failed: " + std::string(e.what()));
        }
    });

    // Pooling operation
    m.def("vulkan_pooling", [checkSize](py::array_t<float> input, py::array_t<float> output,
                                       uint32_t width, uint32_t height, uint32_t depth,
                                       uint32_t poolSizeX, uint32_t poolSizeY, uint32_t strideX, uint32_t strideY) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            uint32_t size_input = checkSize(buf_input.size);
            uint32_t size_output = checkSize(buf_output.size);

            if (poolSizeX == 0 || poolSizeY == 0 || strideX == 0 || strideY == 0) {
                throw std::runtime_error("Pooling sizes and strides must be greater than zero");
            }

            uint32_t expected_output_width = (width - poolSizeX) / strideX + 1;
            uint32_t expected_output_height = (height - poolSizeY) / strideY + 1;
            uint32_t expected_output_size = expected_output_width * expected_output_height * depth;

            if (size_input != width * height * depth) {
                throw std::runtime_error(
                    "Input dimensions mismatch. Expected size: " + 
                    std::to_string(width * height * depth) +
                    ", Got: " + std::to_string(size_input)
                );
            }

            if (size_output != expected_output_size) {
                throw std::runtime_error(
                    "Output dimensions mismatch. Expected size: " + 
                    std::to_string(expected_output_size) +
                    ", Got: " + std::to_string(size_output)
                );
            }

            VulkanTensor tensorInput(size_input * sizeof(float), width, height, depth);
            VulkanTensor tensorOutput(size_output * sizeof(float), expected_output_width, expected_output_height, depth);

            tensorInput.upload(buf_input.ptr);
            executeMaxPool(tensorInput, tensorOutput, width, height, depth, poolSizeX, poolSizeY, strideX, strideY);
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Pooling operation failed: " + std::string(e.what()));
        }
    });
}
