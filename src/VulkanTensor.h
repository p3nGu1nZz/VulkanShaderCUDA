#ifndef VULKAN_TENSOR_H
#define VULKAN_TENSOR_H

#include <vulkan/vulkan.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstring>
#include "VulkanMemoryManager.h"
#include "PushConstants.h"
#include "Utils.h"
#include "VulkanBufferPool.h"
#include "PipelineManager.h"
#include "CommandBufferManager.h"
#include "vulkan_globals.h"

// Tensor Layout Namespace
namespace TensorLayout {
    enum class Layout {
        NHWC,  // Batch, Height, Width, Channels
        NCHW,  // Batch, Channels, Height, Width
        LINEAR // Flat layout
    };

    struct Dimensions {
        uint32_t n;         // Batch size
        uint32_t h;         // Height
        uint32_t w;         // Width
        uint32_t c;         // Channels
        Layout layout;      // Data layout
    };
}

// Vulkan Sync Namespace
namespace VulkanSync {
    class ScopedGPUWait {
    private:
        VkFence fence;
        VkDevice deviceRef;

    public:
        ScopedGPUWait(VkDevice device) : deviceRef(device) {
            VkFenceCreateInfo fenceInfo = {};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = 0;

            VkResult result = vkCreateFence(deviceRef, &fenceInfo, nullptr, &fence);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to create fence during ScopedGPUWait.");
            }
        }

        ~ScopedGPUWait() {
            if (fence != VK_NULL_HANDLE) {
                vkDestroyFence(deviceRef, fence, nullptr);
            }
        }

        VkFence get() const { return fence; }

        void wait() const {
            VkResult result = vkWaitForFences(deviceRef, 1, &fence, VK_TRUE, UINT64_MAX);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to wait for fence in ScopedGPUWait.");
            }
        }

        // Delete copy constructors
        ScopedGPUWait(const ScopedGPUWait&) = delete;
        ScopedGPUWait& operator=(const ScopedGPUWait&) = delete;
    };
};

class VulkanTensor {
private:
    VulkanMemoryManager::AllocationInfo allocation; // Memory allocation info
    TensorLayout::Dimensions dimensions;            // Tensor dimensions
    VkBuffer buffer;                                // Vulkan buffer handle
    VulkanBufferPool* bufferPoolPtr;                // Pointer to the buffer pool
    VkDevice deviceRef;                             // Vulkan device reference

public:
    // Default constructor for pybind11
    VulkanTensor()
        : allocation{nullptr, 0, 0}, dimensions{0, 0, 0, 0, TensorLayout::Layout::LINEAR},
          buffer(VK_NULL_HANDLE), bufferPoolPtr(nullptr), deviceRef(vulkan_globals::device) {}

    VulkanTensor(VulkanMemoryManager* memoryManager, 
                 VulkanBufferPool* bufferPool, 
                 VkDeviceSize size,
                 uint32_t w = 1, 
                 uint32_t h = 1, 
                 uint32_t c = 1, 
                 uint32_t n = 1,
                 const void* data = nullptr, 
                 TensorLayout::Layout layout = TensorLayout::Layout::LINEAR)
        : dimensions{n, h, w, c, layout}
        , bufferPoolPtr(bufferPool)
        , deviceRef(vulkan_globals::device) {
        
        if (deviceRef == VK_NULL_HANDLE) {
            throw std::runtime_error("Vulkan device not initialized.");
        }

        // Allocate memory
        allocation = memoryManager->allocate(
            size, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        
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

        // Bind buffer memory
        result = vkBindBufferMemory(deviceRef, buffer, allocation.memory, allocation.offset);
        if (result != VK_SUCCESS) {
            vkDestroyBuffer(deviceRef, buffer, nullptr);
            throw std::runtime_error("Failed to bind buffer memory in VulkanTensor.");
        }

        // Upload data if provided
        if (data) {
            upload(data);
        }
    }

    // Move constructor
    VulkanTensor(VulkanTensor&& other) noexcept
        : allocation(other.allocation)
        , dimensions(other.dimensions)
        , buffer(other.buffer)
        , bufferPoolPtr(other.bufferPoolPtr)
        , deviceRef(other.deviceRef) {
        other.buffer = VK_NULL_HANDLE;
        other.bufferPoolPtr = nullptr;
    }

    // Move assignment operator
    VulkanTensor& operator=(VulkanTensor&& other) noexcept {
        if (this != &other) {
            cleanup();

            allocation = other.allocation;
            dimensions = other.dimensions;
            buffer = other.buffer;
            bufferPoolPtr = other.bufferPoolPtr;
            deviceRef = other.deviceRef;

            other.buffer = VK_NULL_HANDLE;
            other.bufferPoolPtr = nullptr;
        }
        return *this;
    }

    // Destructor
    ~VulkanTensor() {
        cleanup();
    }

    // Delete copy constructor and assignment
    VulkanTensor(const VulkanTensor&) = delete;
    VulkanTensor& operator=(const VulkanTensor&) = delete;

    // Upload data to Vulkan buffer
    void upload(const void* data) {
        if (!data) {
            throw std::runtime_error("Null pointer provided for upload.");
        }

        void* mappedMemory;
        VkResult result = vkMapMemory(deviceRef, allocation.memory, allocation.offset, allocation.size, 0, &mappedMemory);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to map memory for upload in VulkanTensor.");
        }

        memcpy(mappedMemory, data, static_cast<size_t>(allocation.size));

        VkMappedMemoryRange memoryRange = {};
        memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        memoryRange.memory = allocation.memory;
        memoryRange.offset = allocation.offset;
        memoryRange.size = allocation.size;
        vkFlushMappedMemoryRanges(deviceRef, 1, &memoryRange);

        vkUnmapMemory(deviceRef, allocation.memory);
    }

    // Download data from Vulkan buffer
    void download(void* data) const {
        if (!data) {
            throw std::runtime_error("Null pointer provided for download.");
        }

        void* mappedMemory;
        VkResult result = vkMapMemory(deviceRef, allocation.memory, allocation.offset, allocation.size, 0, &mappedMemory);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to map memory for download in VulkanTensor.");
        }

        VkMappedMemoryRange memoryRange = {};
        memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        memoryRange.memory = allocation.memory;
        memoryRange.offset = allocation.offset;
        memoryRange.size = allocation.size;
        vkInvalidateMappedMemoryRanges(deviceRef, 1, &memoryRange);

        memcpy(data, mappedMemory, static_cast<size_t>(allocation.size));
        vkUnmapMemory(deviceRef, allocation.memory);
    }

    // Getters
    uint32_t getN() const { return dimensions.n; }
    uint32_t getC() const { return dimensions.c; }
    uint32_t getH() const { return dimensions.h; }
    uint32_t getW() const { return dimensions.w; }
    uint32_t getWidth() const { return dimensions.w; }
    uint32_t getHeight() const { return dimensions.h; }
    uint32_t getChannels() const { return dimensions.c; }
    VkDeviceSize getSize() const { return allocation.size; }
    VkBuffer getBuffer() const { return buffer; }

    // Layout Getters and Setters
    TensorLayout::Layout getLayout() const { return dimensions.layout; }
    void setLayout(TensorLayout::Layout newLayout) { dimensions.layout = newLayout; }

    // Utility methods
    bool isValid() const {
        return buffer != VK_NULL_HANDLE && allocation.memory != VK_NULL_HANDLE;
    }

    std::string getDimensionsString() const {
        return "(" + std::to_string(dimensions.n) + ", " +
               std::to_string(dimensions.c) + ", " +
               std::to_string(dimensions.h) + ", " +
               std::to_string(dimensions.w) + ")";
    }

    bool verifyDimensions(VkDeviceSize expectedSize) const {
        return allocation.size == expectedSize &&
               (dimensions.w * dimensions.h * dimensions.c * dimensions.n * sizeof(float)) == expectedSize;
    }

private:
    void cleanup() {
        if (buffer != VK_NULL_HANDLE) {
            if (bufferPoolPtr) {
                bufferPoolPtr->releaseBuffer(buffer);
            }
            vkDestroyBuffer(deviceRef, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
    }
};

#endif // VULKAN_TENSOR_H
