#include <vulkan/vulkan.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

// Vulkan Core Globals
VkInstance instance = VK_NULL_HANDLE;
VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
VkDevice device = VK_NULL_HANDLE;
VkQueue computeQueue = VK_NULL_HANDLE;
VkCommandPool commandPool = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

// Helper Macro for Vulkan Errors
#define VK_CHECK(result)                                    \
    if (result != VK_SUCCESS) {                            \
        throw std::runtime_error("Vulkan Error: " + std::to_string(result)); \
    }

// Initialize Vulkan
void initVulkan() {
    if (instance != VK_NULL_HANDLE) {
        std::cout << "Vulkan already initialized." << std::endl;
        return;
    }

    // Application info
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Backend";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    // Instance creation
    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;

    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance));

    // Enumerate physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan-capable GPUs found.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    physicalDevice = devices[0];

    // Queue and device creation
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = 0; // Assume compute family index is 0
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueCreateInfo;

    VK_CHECK(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device));
    vkGetDeviceQueue(device, 0, 0, &computeQueue);

    // Command pool creation
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;

    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));

    // Descriptor pool creation
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 100;

    VkDescriptorPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &poolSize;
    poolCreateInfo.maxSets = 100;

    VK_CHECK(vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool));

    std::cout << "Vulkan initialized successfully!" << std::endl;
}

// Load SPIR-V Shader
VkShaderModule loadShaderModule(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = buffer.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t *>(buffer.data());

    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule));

    return shaderModule;
}

// Vulkan Tensor Class
class VulkanTensor {
public:
    VulkanTensor(size_t size, uint32_t width = 1, uint32_t height = 1, uint32_t depth = 1, 
                 VkDevice device = nullptr, VkPhysicalDevice physicalDevice = nullptr)
        : size(size), width(width), height(height), depth(depth), device(device), physicalDevice(physicalDevice) {
        if (!device || !physicalDevice) {
            throw std::invalid_argument("Device and PhysicalDevice must not be null.");
        }

        bufferMemory = VK_NULL_HANDLE;
        buffer = createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              bufferMemory);
    }

    ~VulkanTensor() {
        cleanup();
    }

    VulkanTensor(VulkanTensor&& other) noexcept
        : size(other.size), width(other.width), height(other.height), depth(other.depth),
          device(other.device), physicalDevice(other.physicalDevice),
          buffer(other.buffer), bufferMemory(other.bufferMemory) {
        other.buffer = VK_NULL_HANDLE;
        other.bufferMemory = VK_NULL_HANDLE;
    }

    VulkanTensor& operator=(VulkanTensor&& other) noexcept {
        if (this != &other) {
            cleanup();
            size = other.size;
            width = other.width;
            height = other.height;
            depth = other.depth;
            device = other.device;
            physicalDevice = other.physicalDevice;
            buffer = other.buffer;
            bufferMemory = other.bufferMemory;

            other.buffer = VK_NULL_HANDLE;
            other.bufferMemory = VK_NULL_HANDLE;
        }
        return *this;
    }

    VulkanTensor(const VulkanTensor&) = delete;
    VulkanTensor& operator=(const VulkanTensor&) = delete;

    size_t getSize() const { return size; }
    VkBuffer getBuffer() const { return buffer; }
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }
    uint32_t getDepth() const { return depth; }

    void upload(const void* data) {
        void* mappedMemory;
        VK_CHECK(vkMapMemory(device, bufferMemory, 0, size, 0, &mappedMemory));
        memcpy(mappedMemory, data, size);
        vkUnmapMemory(device, bufferMemory);
    }

    void download(void* data) const {
        void* mappedMemory;
        VK_CHECK(vkMapMemory(device, bufferMemory, 0, size, 0, &mappedMemory));
        memcpy(data, mappedMemory, size);
        vkUnmapMemory(device, bufferMemory);
    }

private:
    size_t size;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    VkBuffer createBuffer(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                          VkDeviceMemory& bufferMemory) {
        VkBuffer buffer;

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory));
        vkBindBufferMemory(device, buffer, bufferMemory, 0);

        return buffer;
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type.");
    }

    void cleanup() {
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (bufferMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, bufferMemory, nullptr);
            bufferMemory = VK_NULL_HANDLE;
        }
    }
};

// Execute a compute shader operation
void executeShader(const std::string &shaderPath, 
                  const VulkanTensor &inputA, 
                  const VulkanTensor &inputB, 
                  VulkanTensor &output,
                  uint32_t workgroupSizeX = 256, 
                  uint32_t workgroupSizeY = 1, 
                  uint32_t workgroupSizeZ = 1,
                  bool useThirdBuffer = true) {
    VkShaderModule shaderModule = loadShaderModule(shaderPath);

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
    VK_CHECK(vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPipelineLayout pipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));

    VkDescriptorPool localDescriptorPool;
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = bindingCount;

    VkDescriptorPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &poolSize;
    poolCreateInfo.maxSets = 1;

    VK_CHECK(vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &localDescriptorPool));

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = localDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

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

    vkUpdateDescriptorSets(device, bindingCount, descriptorWrites.data(), 0, nullptr);

    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo commandBufferAllocInfo = {};
    commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocInfo.commandPool = commandPool;
    commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocInfo.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(device, &commandBufferAllocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    uint32_t dispatchX = (inputA.getWidth() + workgroupSizeX - 1) / workgroupSizeX;
    uint32_t dispatchY = (inputA.getHeight() + workgroupSizeY - 1) / workgroupSizeY;
    uint32_t dispatchZ = (inputA.getDepth() + workgroupSizeZ - 1) / workgroupSizeZ;
    vkCmdDispatch(commandBuffer, dispatchX, dispatchY, dispatchZ);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(computeQueue));

    // Cleanup
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyDescriptorPool(device, localDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

// Pybind11 Bindings
PYBIND11_MAKE_OPAQUE(VkDevice);
PYBIND11_MAKE_OPAQUE(VkPhysicalDevice);

// Custom handle wrapper for Vulkan devices
class VulkanDeviceHandle {
public:
    VkDevice device;
    VulkanDeviceHandle() : device(::device) {}
    operator VkDevice() const { return device; }
};

class VulkanPhysicalDeviceHandle {
public:
    VkPhysicalDevice physDevice;
    VulkanPhysicalDeviceHandle() : physDevice(::physicalDevice) {}
    operator VkPhysicalDevice() const { return physDevice; }
};

PYBIND11_MODULE(vulkan_backend, m) {
    m.doc() = "Python bindings for Vulkan operations using Pybind11";

    m.def("init_vulkan", &initVulkan, "Initialize Vulkan backend");

    // Bind the wrapper classes
    py::class_<VulkanDeviceHandle>(m, "VkDevice")
        .def(py::init<>());

    py::class_<VulkanPhysicalDeviceHandle>(m, "VkPhysicalDevice")
        .def(py::init<>());

    // Modify VulkanTensor to use the wrapper classes
    py::class_<VulkanTensor>(m, "VulkanTensor")
        .def(py::init([](size_t size, uint32_t width, uint32_t height, uint32_t depth) {
            if (!device || !physicalDevice) {
                throw std::runtime_error("Vulkan not initialized. Call init_vulkan() first.");
            }
            return std::make_unique<VulkanTensor>(size, width, height, depth, device, physicalDevice);
        }), 
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
                throw std::runtime_error("Data must be a 1D array.");
            }
            if (buf.size * sizeof(float) != self.getSize()) {
                throw std::runtime_error("Data size does not match tensor size.");
            }
            self.upload(buf.ptr);
        })
        .def("download", [](VulkanTensor &self) {
            py::array_t<float> result(self.getSize() / sizeof(float));
            py::buffer_info buf = result.request();
            self.download(buf.ptr);
            return result;
        });
		
		
    m.def("vulkan_add", [](py::array_t<float> a, py::array_t<float> b, py::array_t<float> c) {
        auto buf_a = a.request();
        auto buf_b = b.request();
        auto buf_c = c.request();

        if (buf_a.size != buf_b.size || buf_a.size != buf_c.size) {
            throw std::runtime_error("Input and output arrays must have the same size.");
        }

        VulkanTensor tensorA(buf_a.size * sizeof(float), 1, 1, 1, device, physicalDevice);
        VulkanTensor tensorB(buf_b.size * sizeof(float), 1, 1, 1, device, physicalDevice);
        VulkanTensor tensorC(buf_c.size * sizeof(float), 1, 1, 1, device, physicalDevice);

        tensorA.upload(buf_a.ptr);
        tensorB.upload(buf_b.ptr);

        executeShader("C:\\projects\\VulkanShaderCUDA\\add.spv", tensorA, tensorB, tensorC);

        tensorC.download(buf_c.ptr);
    }, "Perform Vulkan addition");

    m.def("vulkan_pooling", [](py::array_t<float> input, py::array_t<float> output,
                              uint32_t width, uint32_t height, uint32_t depth,
                              uint32_t poolSizeX, uint32_t poolSizeY) {
        auto buf_input = input.request();
        auto buf_output = output.request();

        if (buf_input.size != width * height * depth || 
            buf_output.size != (width / poolSizeX) * (height / poolSizeY) * depth) {
            throw std::runtime_error("Input and output dimensions do not match pooling requirements.");
        }

        VulkanTensor tensorInput(buf_input.size * sizeof(float), width, height, depth, device, physicalDevice);
        VulkanTensor tensorOutput(buf_output.size * sizeof(float), width / poolSizeX, height / poolSizeY, depth, device, physicalDevice);
        // Create a dummy tensor for the second input
        VulkanTensor dummyTensor(sizeof(float), 1, 1, 1, device, physicalDevice);

        tensorInput.upload(buf_input.ptr);

        executeShader("C:\\projects\\VulkanShaderCUDA\\pooling.spv", 
                     tensorInput, dummyTensor, tensorOutput,
                     poolSizeX, poolSizeY, 1, false);

        tensorOutput.download(buf_output.ptr);
    }, "Perform Vulkan pooling operation");

    m.def("vulkan_relu", [](py::array_t<float> input, py::array_t<float> output, uint32_t size) {
        auto buf_input = input.request();
        auto buf_output = output.request();

        if (buf_input.size != buf_output.size || buf_input.size != size) {
            throw std::runtime_error("Input and output sizes must match for ReLU.");
        }

        VulkanTensor tensorInput(buf_input.size * sizeof(float), size, 1, 1, device, physicalDevice);
        VulkanTensor tensorOutput(buf_output.size * sizeof(float), size, 1, 1, device, physicalDevice);
        VulkanTensor dummyTensor(sizeof(float), 1, 1, 1, device, physicalDevice);

        tensorInput.upload(buf_input.ptr);

        executeShader("C:\\projects\\VulkanShaderCUDA\\relu.spv", 
                     tensorInput, dummyTensor, tensorOutput,
                     256, 1, 1, false);

        tensorOutput.download(buf_output.ptr);
    }, "Perform Vulkan ReLU operation");

    m.def("vulkan_sigmoid", [](py::array_t<float> input, py::array_t<float> output, uint32_t size) {
        auto buf_input = input.request();
        auto buf_output = output.request();

        if (buf_input.size != buf_output.size || buf_input.size != size) {
            throw std::runtime_error("Input and output sizes must match for Sigmoid.");
        }

        VulkanTensor tensorInput(buf_input.size * sizeof(float), size, 1, 1, device, physicalDevice);
        VulkanTensor tensorOutput(buf_output.size * sizeof(float), size, 1, 1, device, physicalDevice);
        VulkanTensor dummyTensor(sizeof(float), 1, 1, 1, device, physicalDevice);

        tensorInput.upload(buf_input.ptr);

        executeShader("C:\\projects\\VulkanShaderCUDA\\sigmoid.spv", 
                     tensorInput, dummyTensor, tensorOutput,
                     256, 1, 1, false);

        tensorOutput.download(buf_output.ptr);
    }, "Perform Vulkan Sigmoid operation");

    m.def("vulkan_softmax", [](py::array_t<float> input, py::array_t<float> output, uint32_t size) {
        auto buf_input = input.request();
        auto buf_output = output.request();

        if (buf_input.size != buf_output.size || buf_input.size != size) {
            throw std::runtime_error("Input and output sizes must match for Softmax.");
        }

        VulkanTensor tensorInput(buf_input.size * sizeof(float), size, 1, 1, device, physicalDevice);
        VulkanTensor tensorOutput(buf_output.size * sizeof(float), size, 1, 1, device, physicalDevice);
        VulkanTensor dummyTensor(sizeof(float), 1, 1, 1, device, physicalDevice);

        tensorInput.upload(buf_input.ptr);

        executeShader("C:\\projects\\VulkanShaderCUDA\\softmax.spv", 
                     tensorInput, dummyTensor, tensorOutput,
                     256, 1, 1, false);

        tensorOutput.download(buf_output.ptr);
    }, "Perform Vulkan Softmax operation");

    m.def("vulkan_conv2d", [](py::array_t<float> input, py::array_t<float> kernel, py::array_t<float> output,
                              uint32_t inputWidth, uint32_t inputHeight, uint32_t inputDepth,
                              uint32_t kernelWidth, uint32_t kernelHeight, uint32_t outputDepth) {
        auto buf_input = input.request();
        auto buf_kernel = kernel.request();
        auto buf_output = output.request();

        if (buf_output.size != inputWidth * inputHeight * outputDepth) {
            throw std::runtime_error("Output dimensions do not match Conv2D requirements.");
        }

        VulkanTensor tensorInput(buf_input.size * sizeof(float), inputWidth, inputHeight, inputDepth, device, physicalDevice);
        VulkanTensor tensorKernel(buf_kernel.size * sizeof(float), kernelWidth, kernelHeight, inputDepth, device, physicalDevice);
        VulkanTensor tensorOutput(buf_output.size * sizeof(float), inputWidth, inputHeight, outputDepth, device, physicalDevice);

        tensorInput.upload(buf_input.ptr);
        tensorKernel.upload(buf_kernel.ptr);

        executeShader("C:\\projects\\VulkanShaderCUDA\\conv2d.spv", 
                     tensorInput, tensorKernel, tensorOutput,
                     16, 16, 1);

        tensorOutput.download(buf_output.ptr);
    }, "Perform Vulkan Conv2D operation");

    m.def("vulkan_mul", [](py::array_t<float> inputA, py::array_t<float> inputB, py::array_t<float> output) {
        auto buf_a = inputA.request();
        auto buf_b = inputB.request();
        auto buf_c = output.request();

        if (buf_a.size != buf_b.size || buf_a.size != buf_c.size) {
            throw std::runtime_error("Input and output sizes must match for Multiplication.");
        }

        VulkanTensor tensorA(buf_a.size * sizeof(float), 1, 1, 1, device, physicalDevice);
        VulkanTensor tensorB(buf_b.size * sizeof(float), 1, 1, 1, device, physicalDevice);
        VulkanTensor tensorC(buf_c.size * sizeof(float), 1, 1, 1, device, physicalDevice);

        tensorA.upload(buf_a.ptr);
        tensorB.upload(buf_b.ptr);

        executeShader("C:\\projects\\VulkanShaderCUDA\\mul.spv", tensorA, tensorB, tensorC);

        tensorC.download(buf_c.ptr);
    }, "Perform Vulkan element-wise multiplication");
}