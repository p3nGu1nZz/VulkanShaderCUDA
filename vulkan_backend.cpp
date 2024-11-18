#include <vulkan/vulkan.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

// Vulkan Core Globals
VkInstance instance;
VkPhysicalDevice physicalDevice;
VkDevice device;
VkQueue computeQueue;
VkCommandPool commandPool;
VkDescriptorPool descriptorPool;

// Helper Macro for Vulkan Errors
#define VK_CHECK(result)                                    \
    if (result != VK_SUCCESS) {                            \
        throw std::runtime_error("Vulkan Error: " + std::to_string(result)); \
    }

// Initialize Vulkan
void initVulkan() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Torch Backend";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;

    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance));

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan-capable GPUs found.");
    }
    VkPhysicalDevice devices[10];
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
    physicalDevice = devices[0];

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = 0; // Assuming compute family index is 0
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueCreateInfo;

    VK_CHECK(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device));
    vkGetDeviceQueue(device, 0, 0, &computeQueue);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;

    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));

    // Descriptor Pool for Buffers
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 100; // Arbitrary large number for now

    VkDescriptorPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &poolSize;
    poolCreateInfo.maxSets = 100; // Arbitrary large number for now

    VK_CHECK(vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool));

    std::cout << "Vulkan initialized successfully!" << std::endl;
}

// Load SPIR-V Shader
std::vector<char> loadShader(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file.");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

// Vulkan Tensor Class
class VulkanTensor {
public:
    VulkanTensor(size_t size) : size(size) {
        bufferMemory = VK_NULL_HANDLE;
        buffer = createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              bufferMemory);
    }

    ~VulkanTensor() {
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, bufferMemory, nullptr);
    }

    size_t getSize() const { return size; }
    VkBuffer getBuffer() const { return buffer; }

    void upload(const void *data) {
        void *mappedMemory;
        vkMapMemory(device, bufferMemory, 0, size, 0, &mappedMemory);
        memcpy(mappedMemory, data, size);
        vkUnmapMemory(device, bufferMemory);
    }

    void download(void *data) const {
        void *mappedMemory;
        vkMapMemory(device, bufferMemory, 0, size, 0, &mappedMemory);
        memcpy(data, mappedMemory, size);
        vkUnmapMemory(device, bufferMemory);
    }

private:
    size_t size;
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;

    VkBuffer createBuffer(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                          VkDeviceMemory &bufferMemory) {
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

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type.");
    }
};

// Create Compute Pipeline
VkPipeline createComputePipeline(VkShaderModule shaderModule, VkPipelineLayout pipelineLayout) {
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
    return pipeline;
}

// Pybind11 Bindings
PYBIND11_MODULE(vulkan_backend, m) {
    m.def("init_vulkan", &initVulkan, "Initialize Vulkan backend");

    py::class_<VulkanTensor>(m, "VulkanTensor")
        .def(py::init<size_t>())
        .def("get_size", &VulkanTensor::getSize)
        .def("upload", [](VulkanTensor &self, py::array_t<float> data) {
            py::buffer_info buf = data.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Data must be a 1D array");
            }
            self.upload(buf.ptr);
        })
        .def("download", [](VulkanTensor &self) {
            py::array_t<float> result(self.getSize() / 4);
            py::buffer_info buf = result.request();
            self.download(buf.ptr);
            return result;
        });
}
