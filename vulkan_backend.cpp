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
        size_t size;
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

    std::pair<VkBuffer, VkDeviceMemory> acquireBuffer(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        
        for (auto& info : buffers) {
            if (!info.in_use && info.size >= size) {
                info.in_use = true;
                return {info.buffer, info.memory};
            }
        }

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
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
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::unique_ptr<VulkanBufferPool> bufferPool;

    ~VulkanContext() {
        if (descriptorPool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
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
    size_t size;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDevice device;
    VulkanBufferPool* bufferPool;

public:
    VulkanTensor(size_t size, uint32_t w = 1, uint32_t h = 1, uint32_t d = 1)
        : size(size), width(w), height(h), depth(d) {
        if (!vulkanContext || !vulkanContext->device) {
            throw std::runtime_error("Vulkan not initialized");
        }
        device = vulkanContext->device;
        bufferPool = vulkanContext->bufferPool.get();
        
        auto bufferPair = bufferPool->acquireBuffer(size);
        buffer = bufferPair.first;
        memory = bufferPair.second;
    }

    ~VulkanTensor() {
        if (buffer != VK_NULL_HANDLE) {
            bufferPool->releaseBuffer(buffer);
        }
    }

    VulkanTensor(VulkanTensor&& other) noexcept
        : size(other.size), width(other.width), height(other.height), depth(other.depth),
          buffer(other.buffer), memory(other.memory), device(other.device),
          bufferPool(other.bufferPool) {
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
        memcpy(mappedMemory, data, size);
        vkUnmapMemory(device, memory);
    }

    void download(void* data) const {
        void* mappedMemory;
        VK_CHECK_DETAILED(
            vkMapMemory(device, memory, 0, size, 0, &mappedMemory),
            "Memory Mapping for Download"
        );
        memcpy(data, mappedMemory, size);
        vkUnmapMemory(device, memory);
    }

    size_t getSize() const { return size; }
    VkBuffer getBuffer() const { return buffer; }
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }
    uint32_t getDepth() const { return depth; }
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

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = 0;
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

    vkGetDeviceQueue(vulkanContext->device, 0, 0, &vulkanContext->computeQueue);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
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
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
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

// Enhanced compute shader execution
void executeShader(const std::string& shaderName,
                  const VulkanTensor& inputA,
                  const VulkanTensor& inputB,
                  VulkanTensor& output,
                  uint32_t workgroupSizeX = 256,
                  uint32_t workgroupSizeY = 1,
                  uint32_t workgroupSizeZ = 1,
                  bool useThirdBuffer = true) {
    
    std::string shaderPath = getShaderPath(shaderName);
    auto shaderModule = loadShaderModule(shaderPath);
    std::unique_ptr<VulkanResource> shaderResource(
        new VulkanResource(
            vulkanContext->device,
            [device = vulkanContext->device, shaderModule]() {
                vkDestroyShaderModule(device, shaderModule, nullptr);
            }
        )
    );

    // Create pipeline layout with push constants
    const int bindingCount = useThirdBuffer ? 3 : 2;
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

    VkDescriptorSetLayout descriptorSetLayout;
    VK_CHECK_DETAILED(
        vkCreateDescriptorSetLayout(vulkanContext->device, &descriptorLayoutInfo, nullptr, &descriptorSetLayout),
        "Descriptor Set Layout Creation"
    );

    // Push constant range
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(MatMulPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = shaderName == "shaders/matmul.spv" ? 1 : 0;
    pipelineLayoutInfo.pPushConstantRanges = shaderName == "shaders/matmul.spv" ? &pushConstantRange : nullptr;

    VkPipelineLayout pipelineLayout;
    VK_CHECK_DETAILED(
        vkCreatePipelineLayout(vulkanContext->device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
        "Pipeline Layout Creation"
    );

    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    VK_CHECK_DETAILED(
        vkCreateComputePipelines(vulkanContext->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
        "Compute Pipeline Creation"
    );

    // Set up descriptor pool and sets
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = bindingCount;

    VkDescriptorPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &poolSize;
    poolCreateInfo.maxSets = 1;

    VkDescriptorPool descriptorPool;
    VK_CHECK_DETAILED(
        vkCreateDescriptorPool(vulkanContext->device, &poolCreateInfo, nullptr, &descriptorPool),
        "Descriptor Pool Creation"
    );

    // Allocate descriptor sets
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    VK_CHECK_DETAILED(
        vkAllocateDescriptorSets(vulkanContext->device, &allocInfo, &descriptorSet),
        "Descriptor Set Allocation"
    );

    // Update descriptor sets
    std::vector<VkDescriptorBufferInfo> bufferInfos(bindingCount);
    std::vector<VkWriteDescriptorSet> descriptorWrites(bindingCount);

    bufferInfos[0] = {inputA.getBuffer(), 0, inputA.getSize()};
    bufferInfos[1] = {inputB.getBuffer(), 0, inputB.getSize()};
    if (useThirdBuffer) {
        bufferInfos[2] = {output.getBuffer(), 0, output.getSize()};
    }

    for (int i = 0; i < bindingCount; ++i) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }

    vkUpdateDescriptorSets(vulkanContext->device, bindingCount, descriptorWrites.data(), 0, nullptr);

    // Record command buffer
    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo commandBufferAllocInfo = {};
    commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocInfo.commandPool = vulkanContext->commandPool;
    commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocInfo.commandBufferCount = 1;

    VK_CHECK_DETAILED(
        vkAllocateCommandBuffers(vulkanContext->device, &commandBufferAllocInfo, &commandBuffer),
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

    // Handle matmul push constants and dispatch dimensions
    uint32_t dispatchX, dispatchY, dispatchZ;
    if (shaderName == "shaders/matmul.spv") {
        MatMulPushConstants constants{
            inputA.getHeight(),
            inputA.getWidth(),
            inputB.getWidth()
        };
        
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(MatMulPushConstants), &constants);

        dispatchX = (inputB.getWidth() + workgroupSizeX - 1) / workgroupSizeX;
        dispatchY = (inputA.getHeight() + workgroupSizeY - 1) / workgroupSizeY;
        dispatchZ = 1;
    } else {
        dispatchX = (inputA.getWidth() + workgroupSizeX - 1) / workgroupSizeX;
        dispatchY = (inputA.getHeight() + workgroupSizeY - 1) / workgroupSizeY;
        dispatchZ = (inputA.getDepth() + workgroupSizeZ - 1) / workgroupSizeZ;
    }

    vkCmdDispatch(commandBuffer, dispatchX, dispatchY, dispatchZ);

    VK_CHECK_DETAILED(
        vkEndCommandBuffer(commandBuffer),
        "Command Buffer Recording End"
    );

    // Submit and wait
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

    // Cleanup
    vkFreeCommandBuffers(vulkanContext->device, vulkanContext->commandPool, 1, &commandBuffer);
    vkDestroyPipeline(vulkanContext->device, pipeline, nullptr);
    vkDestroyPipelineLayout(vulkanContext->device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(vulkanContext->device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(vulkanContext->device, descriptorPool, nullptr);
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
        .def(py::init<size_t, uint32_t, uint32_t, uint32_t>(),
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
            size_t expected_size = self.getSize();
            size_t actual_size = buf.size * sizeof(float);
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

    // Matrix multiplication
    m.def("vulkan_matmul", [checkSize](py::array_t<float> inputA, py::array_t<float> inputB, py::array_t<float> output) {
        try {
            auto buf_a = inputA.request();
            auto buf_b = inputB.request();
            auto buf_c = output.request();

            if (buf_a.ndim != 2 || buf_b.ndim != 2) {
                throw std::runtime_error("Inputs must be 2D matrices");
            }

            uint32_t M = checkSize(buf_a.shape[0]); // Height of A
            uint32_t K = checkSize(buf_a.shape[1]); // Width of A
            uint32_t N = checkSize(buf_b.shape[1]); // Width of B

            if (buf_b.shape[0] != K) {
                throw std::runtime_error(
                    "Matrix dimensions mismatch. A: " + std::to_string(M) + "x" + std::to_string(K) +
                    ", B: " + std::to_string(buf_b.shape[0]) + "x" + std::to_string(N)
                );
            }

            if (buf_c.shape[0] != M || buf_c.shape[1] != N) {
                throw std::runtime_error(
                    "Output matrix has wrong dimensions. Expected: " + std::to_string(M) + "x" + std::to_string(N) +
                    ", Got: " + std::to_string(buf_c.shape[0]) + "x" + std::to_string(buf_c.shape[1])
                );
            }

            VulkanTensor tensorA(M * K * sizeof(float), K, M, 1);
            VulkanTensor tensorB(K * N * sizeof(float), N, K, 1);
            VulkanTensor tensorC(M * N * sizeof(float), N, M, 1);

            tensorA.upload(buf_a.ptr);
            tensorB.upload(buf_b.ptr);

            executeShader("shaders/matmul.spv", tensorA, tensorB, tensorC, 16, 16, 1);

            tensorC.download(buf_c.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Matrix multiplication failed: " + std::string(e.what()));
        }
    });

    // ReLU operation
    m.def("vulkan_relu", [checkSize](py::array_t<float> input, py::array_t<float> output, uint32_t size) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            uint32_t input_size = checkSize(buf_input.size);
            uint32_t output_size = checkSize(buf_output.size);

            if (input_size != size || output_size != size) {
                throw std::runtime_error(
                    "Size mismatch. Expected: " + std::to_string(size) +
                    ", Got input: " + std::to_string(input_size) +
                    ", output: " + std::to_string(output_size)
                );
            }

            VulkanTensor tensorInput(input_size * sizeof(float), size, 1, 1);
            VulkanTensor tensorOutput(output_size * sizeof(float), size, 1, 1);
            VulkanTensor dummyTensor(sizeof(float), 1, 1, 1);

            tensorInput.upload(buf_input.ptr);

            executeShader("shaders/relu.spv", tensorInput, dummyTensor, tensorOutput, 256, 1, 1, false);

            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("ReLU operation failed: " + std::string(e.what()));
        }
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

            executeShader("shaders/add.spv", tensorA, tensorB, tensorC);

            tensorC.download(buf_c.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Addition operation failed: " + std::string(e.what()));
        }
    });

    // Sigmoid operation
    m.def("vulkan_sigmoid", [checkSize](py::array_t<float> input, py::array_t<float> output, uint32_t size) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            uint32_t input_size = checkSize(buf_input.size);
            uint32_t output_size = checkSize(buf_output.size);

            if (input_size != size || output_size != size) {
                throw std::runtime_error(
                    "Size mismatch. Expected: " + std::to_string(size) +
                    ", Got input: " + std::to_string(input_size) +
                    ", output: " + std::to_string(output_size)
                );
            }

            VulkanTensor tensorInput(input_size * sizeof(float), size, 1, 1);
            VulkanTensor tensorOutput(output_size * sizeof(float), size, 1, 1);
            VulkanTensor dummyTensor(sizeof(float), 1, 1, 1);

            tensorInput.upload(buf_input.ptr);

            executeShader("shaders/sigmoid.spv", tensorInput, dummyTensor, tensorOutput, 256, 1, 1, false);

            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Sigmoid operation failed: " + std::string(e.what()));
        }
    });

    // Softmax operation
    m.def("vulkan_softmax", [checkSize](py::array_t<float> input, py::array_t<float> output, uint32_t size) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            uint32_t input_size = checkSize(buf_input.size);
            uint32_t output_size = checkSize(buf_output.size);

            if (input_size != size || output_size != size) {
                throw std::runtime_error(
                    "Size mismatch. Expected: " + std::to_string(size) +
                    ", Got input: " + std::to_string(input_size) +
                    ", output: " + std::to_string(output_size)
                );
            }

            VulkanTensor tensorInput(input_size * sizeof(float), size, 1, 1);
            VulkanTensor tensorOutput(output_size * sizeof(float), size, 1, 1);
            VulkanTensor dummyTensor(sizeof(float), 1, 1, 1);

            tensorInput.upload(buf_input.ptr);

            executeShader("shaders/softmax.spv", tensorInput, dummyTensor, tensorOutput, 256, 1, 1, false);

            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Softmax operation failed: " + std::string(e.what()));
        }
    });
	// Add this to the PYBIND11_MODULE section
    m.def("vulkan_pooling", [checkSize](py::array_t<float> input, py::array_t<float> output,
                              uint32_t width, uint32_t height, uint32_t depth,
                              uint32_t poolSizeX, uint32_t poolSizeY) {
        try {
            auto buf_input = input.request();
            auto buf_output = output.request();

            uint32_t input_size = checkSize(buf_input.size);
            uint32_t output_size = checkSize(buf_output.size);

            uint32_t expected_output_width = width / poolSizeX;
            uint32_t expected_output_height = height / poolSizeY;
            uint32_t expected_output_size = expected_output_width * expected_output_height * depth;

            if (input_size != width * height * depth) {
                throw std::runtime_error(
                    "Input dimensions mismatch. Expected size: " + 
                    std::to_string(width * height * depth) +
                    ", Got: " + std::to_string(input_size)
                );
            }

            if (output_size != expected_output_size) {
                throw std::runtime_error(
                    "Output dimensions mismatch. Expected size: " + 
                    std::to_string(expected_output_size) +
                    ", Got: " + std::to_string(output_size)
                );
            }

            VulkanTensor tensorInput(input_size * sizeof(float), width, height, depth);
            VulkanTensor tensorOutput(output_size * sizeof(float), expected_output_width, expected_output_height, depth);
            VulkanTensor dummyTensor(sizeof(float), 1, 1, 1);

            tensorInput.upload(buf_input.ptr);

            executeShader("shaders/pooling.spv", 
                         tensorInput, dummyTensor, tensorOutput,
                         poolSizeX, poolSizeY, 1, false);

            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Pooling operation failed: " + std::string(e.what()));
        }
    });	
	
	m.def("vulkan_conv2d", [checkSize](py::array_t<float> input, py::array_t<float> kernel, py::array_t<float> output,
                              uint32_t inputWidth, uint32_t inputHeight, uint32_t inputDepth,
                              uint32_t kernelWidth, uint32_t kernelHeight, uint32_t outputDepth) {
        try {
            auto buf_input = input.request();
            auto buf_kernel = kernel.request();
            auto buf_output = output.request();

            uint32_t input_size = checkSize(buf_input.size);
            uint32_t kernel_size = checkSize(buf_kernel.size);
            uint32_t output_size = checkSize(buf_output.size);

            // Calculate correct output dimensions
            int outputWidth = inputWidth - kernelWidth + 1;
            int outputHeight = inputHeight - kernelHeight + 1;

            if (output_size != static_cast<uint32_t>(outputWidth * outputHeight * outputDepth)) {
                throw std::runtime_error(
                    "Output dimensions do not match Conv2D requirements. "
                    "Expected: " + std::to_string(outputWidth * outputHeight * outputDepth) +
                    ", Got: " + std::to_string(output_size)
                );
            }

            VulkanTensor tensorInput(input_size * sizeof(float), inputWidth, inputHeight, inputDepth);
            VulkanTensor tensorKernel(kernel_size * sizeof(float), kernelWidth, kernelHeight, inputDepth);
            VulkanTensor tensorOutput(output_size * sizeof(float), outputWidth, outputHeight, outputDepth);

            tensorInput.upload(buf_input.ptr);
            tensorKernel.upload(buf_kernel.ptr);

            executeShader("shaders/conv2d.spv", tensorInput, tensorKernel, tensorOutput, 16, 16, 1);

            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Conv2D operation failed: " + std::string(e.what()));
        }
    });
	
}