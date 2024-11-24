#ifndef VULKAN_CONTEXT_H
#define VULKAN_CONTEXT_H

#include <vulkan/vulkan.h>
#include <memory>
#include "DescriptorSetManager.h"
#include "CommandBufferManager.h"
#include "PipelineManager.h"
#include "VulkanMemoryManager.h"
#include "VulkanBufferPool.h"

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    void initVulkan();
    void cleanupVulkan();

    // Accessors
    VulkanMemoryManager* getMemoryManager() const { return memoryManager.get(); }
    VulkanBufferPool* getBufferPool() const { return bufferPool.get(); }
    PipelineManager* getPipelineManager() const { return pipelineManager.get(); }
    CommandBufferManager* getCommandBufferManager() const { return commandBufferManager.get(); }
    std::shared_ptr<DescriptorSetManager> getDescriptorSetManager() const { return descriptorSetManager; }
    
    // Device accessors
    VkDevice getDevice() const { return device; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }

private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;

    std::unique_ptr<VulkanMemoryManager> memoryManager;
    std::unique_ptr<VulkanBufferPool> bufferPool;
    std::unique_ptr<PipelineManager> pipelineManager;
    std::unique_ptr<CommandBufferManager> commandBufferManager;
    std::shared_ptr<DescriptorSetManager> descriptorSetManager;

    void pickPhysicalDevice();
    void createLogicalDevice();
    void findComputeQueueFamily();
};

#endif // VULKAN_CONTEXT_H