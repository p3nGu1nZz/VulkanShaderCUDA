#include "VulkanTensor.h"
#include "VulkanDeviceHelper.h"
#include "VulkanSync.h"
#include "vulkan_globals.h"
#include <iostream>
#include <algorithm>
#include <spdlog/spdlog.h>

VulkanTensor::VulkanTensor()
    : allocation{VK_NULL_HANDLE, 0, 0}
    , dimensions{0, 0, 0, 0, TensorLayout::Layout::LINEAR}
    , buffer(VK_NULL_HANDLE)
    , bufferPoolPtr(nullptr)
    , deviceRef(vulkan_globals::device) {
    spdlog::debug("Created empty VulkanTensor");
}

VulkanTensor::VulkanTensor(
    VulkanMemoryManager* memoryManager,
    VulkanBufferPool* bufferPool,
    VkDeviceSize size,
    uint32_t w,
    uint32_t h,
    uint32_t c,
    uint32_t n,
    const void* data,
    TensorLayout::Layout layout)
    : allocation{VK_NULL_HANDLE, 0, 0}
    , dimensions{n, h, w, c, layout}
    , buffer(VK_NULL_HANDLE)
    , bufferPoolPtr(bufferPool)
    , deviceRef(vulkan_globals::device) {
    if (!deviceRef) {
        throw std::runtime_error("Vulkan device not initialized.");
    }

    auto context = vulkan_globals::getContext();
    if (!context) {
        throw std::runtime_error("Vulkan context is null.");
    }

    // Get device properties for alignment
    VkPhysicalDeviceProperties deviceProperties = 
        VulkanDeviceHelper::getPhysicalDeviceProperties(context->getPhysicalDevice());
    
    // Ensure size alignment
    VkDeviceSize alignment = deviceProperties.limits.minStorageBufferOffsetAlignment;
    size = (size + alignment - 1) & ~(alignment - 1);

    // Allocate memory
    try {
        allocation = memoryManager->allocate(
            size,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to allocate memory: ") + e.what());
    }

    // Create buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(deviceRef, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer in VulkanTensor.");
    }

    result = vkBindBufferMemory(deviceRef, buffer, allocation.memory, allocation.offset);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(deviceRef, buffer, nullptr);
        throw std::runtime_error("Failed to bind buffer memory in VulkanTensor.");
    }

    spdlog::debug("Created VulkanTensor with dimensions {}x{}x{}x{}", n, h, w, c);

    if (data) {
        upload(data);
    }
}

VulkanTensor::VulkanTensor(VulkanTensor&& other) noexcept
    : allocation(other.allocation)
    , dimensions(other.dimensions)
    , buffer(other.buffer)
    , bufferPoolPtr(other.bufferPoolPtr)
    , deviceRef(other.deviceRef) {
    other.buffer = VK_NULL_HANDLE;
    other.bufferPoolPtr = nullptr;
    spdlog::debug("Moved VulkanTensor");
}

VulkanTensor& VulkanTensor::operator=(VulkanTensor&& other) noexcept {
    if (this != &other) {
        cleanup();

        allocation = other.allocation;
        dimensions = other.dimensions;
        buffer = other.buffer;
        bufferPoolPtr = other.bufferPoolPtr;
        deviceRef = other.deviceRef;

        other.buffer = VK_NULL_HANDLE;
        other.bufferPoolPtr = nullptr;
        spdlog::debug("Move assigned VulkanTensor");
    }
    return *this;
}

VulkanTensor::~VulkanTensor() {
    cleanup();
    spdlog::debug("Destroyed VulkanTensor");
}

void VulkanTensor::upload(const void* data) {
    if (!data) {
        throw std::runtime_error("Null pointer provided for upload.");
    }

    VulkanContext* context = vulkan_globals::getContext();
    if (!context) {
        throw std::runtime_error("Vulkan context is null.");
    }

    VkCommandBuffer cmdBuffer = context->getCommandBufferManager()->acquireCommandBuffer();
    
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    if (vkBeginCommandBuffer(cmdBuffer, &beginInfo) != VK_SUCCESS) {
        context->getCommandBufferManager()->releaseCommandBuffer(cmdBuffer);
        throw std::runtime_error("Failed to begin command buffer during upload.");
    }

    // Add buffer memory barrier for upload
    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = allocation.size;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr
    );

    if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
        context->getCommandBufferManager()->releaseCommandBuffer(cmdBuffer);
        throw std::runtime_error("Failed to end command buffer during upload.");
    }

    // Submit and wait
    {
        VulkanSync::ScopedGPUWait scopedWait(deviceRef);
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        if (vkQueueSubmit(vulkan_globals::computeQueue, 1, &submitInfo, scopedWait.get()) != VK_SUCCESS) {
            context->getCommandBufferManager()->releaseCommandBuffer(cmdBuffer);
            throw std::runtime_error("Failed to submit command buffer during upload.");
        }

        scopedWait.wait();
    }

    // Map and copy data
    void* mappedMemory;
    VkResult result = vkMapMemory(deviceRef, allocation.memory, allocation.offset, allocation.size, 0, &mappedMemory);
    if (result != VK_SUCCESS) {
        context->getCommandBufferManager()->releaseCommandBuffer(cmdBuffer);
        throw std::runtime_error("Failed to map memory for upload.");
    }

    memcpy(mappedMemory, data, static_cast<size_t>(allocation.size));

    VkMappedMemoryRange memoryRange = {};
    memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    memoryRange.memory = allocation.memory;
    memoryRange.offset = allocation.offset;
    memoryRange.size = allocation.size;
    vkFlushMappedMemoryRanges(deviceRef, 1, &memoryRange);

    vkUnmapMemory(deviceRef, allocation.memory);
    context->getCommandBufferManager()->releaseCommandBuffer(cmdBuffer);

    spdlog::debug("Uploaded data to VulkanTensor");
}

void VulkanTensor::download(void* data) const {
    if (!data) {
        throw std::runtime_error("Null pointer provided for download.");
    }

    VulkanContext* context = vulkan_globals::getContext();
    if (!context) {
        throw std::runtime_error("Vulkan context is null.");
    }

    VkCommandBuffer cmdBuffer = context->getCommandBufferManager()->acquireCommandBuffer();
    
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    if (vkBeginCommandBuffer(cmdBuffer, &beginInfo) != VK_SUCCESS) {
        context->getCommandBufferManager()->releaseCommandBuffer(cmdBuffer);
        throw std::runtime_error("Failed to begin command buffer during download.");
    }

    // Add buffer memory barrier for download
    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = allocation.size;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr
    );

    if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
        context->getCommandBufferManager()->releaseCommandBuffer(cmdBuffer);
        throw std::runtime_error("Failed to end command buffer during download.");
    }

    // Submit and wait
    {
        VulkanSync::ScopedGPUWait scopedWait(deviceRef);
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        if (vkQueueSubmit(vulkan_globals::computeQueue, 1, &submitInfo, scopedWait.get()) != VK_SUCCESS) {
            context->getCommandBufferManager()->releaseCommandBuffer(cmdBuffer);
            throw std::runtime_error("Failed to submit command buffer during download.");
        }

        scopedWait.wait();
    }

    // Map and copy data
    void* mappedMemory;
    VkResult result = vkMapMemory(deviceRef, allocation.memory, allocation.offset, allocation.size, 0, &mappedMemory);
    if (result != VK_SUCCESS) {
        context->getCommandBufferManager()->releaseCommandBuffer(cmdBuffer);
        throw std::runtime_error("Failed to map memory for download.");
    }

    VkMappedMemoryRange memoryRange = {};
    memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    memoryRange.memory = allocation.memory;
    memoryRange.offset = allocation.offset;
    memoryRange.size = allocation.size;
    vkInvalidateMappedMemoryRanges(deviceRef, 1, &memoryRange);

    memcpy(data, mappedMemory, static_cast<size_t>(allocation.size));
    vkUnmapMemory(deviceRef, allocation.memory);

    context->getCommandBufferManager()->releaseCommandBuffer(cmdBuffer);
    spdlog::debug("Downloaded data from VulkanTensor");
}

uint32_t VulkanTensor::getN() const { return dimensions.n; }
uint32_t VulkanTensor::getC() const { return dimensions.c; }
uint32_t VulkanTensor::getH() const { return dimensions.h; }
uint32_t VulkanTensor::getW() const { return dimensions.w; }
uint32_t VulkanTensor::getWidth() const { return dimensions.w; }
uint32_t VulkanTensor::getHeight() const { return dimensions.h; }
uint32_t VulkanTensor::getChannels() const { return dimensions.c; }
VkDeviceSize VulkanTensor::getSize() const { return allocation.size; }
VkBuffer VulkanTensor::getBuffer() const { return buffer; }

TensorLayout::Layout VulkanTensor::getLayout() const { return dimensions.layout; }
void VulkanTensor::setLayout(TensorLayout::Layout newLayout) { dimensions.layout = newLayout; }

bool VulkanTensor::isValid() const {
    return buffer != VK_NULL_HANDLE && allocation.memory != VK_NULL_HANDLE;
}

std::string VulkanTensor::getDimensionsString() const {
    return "(" + std::to_string(dimensions.n) + ", " +
           std::to_string(dimensions.c) + ", " +
           std::to_string(dimensions.h) + ", " +
           std::to_string(dimensions.w) + ")";
}

bool VulkanTensor::verifyDimensions(VkDeviceSize expectedSize) const {
    return allocation.size == expectedSize &&
           (dimensions.w * dimensions.h * dimensions.c * dimensions.n * sizeof(float)) == expectedSize;
}

void VulkanTensor::debugPrint() const {
    std::vector<float> data(allocation.size / sizeof(float));
    download(data.data());
    std::cout << "Tensor data (first 10 elements): ";
    for (size_t i = 0; i < std::min<size_t>(10, data.size()); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void VulkanTensor::cleanup() {
    if (buffer != VK_NULL_HANDLE) {
        if (bufferPoolPtr) {
            bufferPoolPtr->releaseBuffer(buffer);
        }
        vkDestroyBuffer(deviceRef, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        spdlog::debug("Cleaned up VulkanTensor resources");
    }
}