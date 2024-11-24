#ifndef VULKAN_TENSOR_H
#define VULKAN_TENSOR_H

#include <vulkan/vulkan.h>
#include <string>
#include "VulkanMemoryManager.h"
#include "VulkanBufferPool.h"
#include "vulkan_globals.h"

namespace TensorLayout {
    enum class Layout {
        NHWC,   // Batch, Height, Width, Channels
        NCHW,   // Batch, Channels, Height, Width
        LINEAR  // Flat layout
    };
}

struct TensorDimensions {
    uint32_t n;
    uint32_t h;
    uint32_t w;
    uint32_t c;
    TensorLayout::Layout layout;
};

class VulkanTensor {
public:
    VulkanTensor();
    
    VulkanTensor(
        VulkanMemoryManager* memoryManager,
        VulkanBufferPool* bufferPool,
        VkDeviceSize size,
        uint32_t w,
        uint32_t h,
        uint32_t c,
        uint32_t n,
        const void* data,
        TensorLayout::Layout layout
    );

    // Move constructor and assignment
    VulkanTensor(VulkanTensor&& other) noexcept;
    VulkanTensor& operator=(VulkanTensor&& other) noexcept;

    // Delete copy operations
    VulkanTensor(const VulkanTensor&) = delete;
    VulkanTensor& operator=(const VulkanTensor&) = delete;

    ~VulkanTensor();

    void upload(const void* data);
    void download(void* data) const;

    // Accessors
    uint32_t getN() const;
    uint32_t getC() const;
    uint32_t getH() const;
    uint32_t getW() const;
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    uint32_t getChannels() const;
    VkDeviceSize getSize() const;
    VkBuffer getBuffer() const;
    TensorLayout::Layout getLayout() const;
    void setLayout(TensorLayout::Layout newLayout);

    bool isValid() const;
    std::string getDimensionsString() const;
    bool verifyDimensions(VkDeviceSize expectedSize) const;
    void debugPrint() const;

private:
    void cleanup();

    VulkanMemoryManager::AllocationInfo allocation;
    TensorDimensions dimensions;
    VkBuffer buffer;
    VulkanBufferPool* bufferPoolPtr;
    VkDevice deviceRef;
};

#endif // VULKAN_TENSOR_H