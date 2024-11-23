#ifndef COMMANDBUFFERMANAGER_H
#define COMMANDBUFFERMANAGER_H

#include <vulkan/vulkan.h>
#include <mutex>
#include <queue>

class CommandBufferManager {
public:
    CommandBufferManager(VkDevice device, VkCommandPool commandPool);
    ~CommandBufferManager();

    VkCommandBuffer acquireCommandBuffer();
    void releaseCommandBuffer(VkCommandBuffer commandBuffer);

private:
    VkDevice device;
    VkCommandPool commandPool;
    std::mutex mutex;
    std::queue<VkCommandBuffer> commandBuffers;
};

#endif // COMMANDBUFFERMANAGER_H
