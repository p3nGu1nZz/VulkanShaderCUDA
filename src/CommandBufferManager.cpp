#include "CommandBufferManager.h"
#include "spdlog/spdlog.h"

CommandBufferManager::CommandBufferManager(VkDevice device, VkCommandPool commandPool)
    : device(device), commandPool(commandPool) {
    spdlog::info("CommandBufferManager created.");
}

CommandBufferManager::~CommandBufferManager() {
    std::lock_guard<std::mutex> lock(mutex);

    while (!commandBuffers.empty()) {
        VkCommandBuffer cmd = commandBuffers.front();
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
        commandBuffers.pop();
    }
    spdlog::info("CommandBufferManager destroyed.");
}

VkCommandBuffer CommandBufferManager::acquireCommandBuffer() {
    std::lock_guard<std::mutex> lock(mutex);

    if (!commandBuffers.empty()) {
        VkCommandBuffer cmd = commandBuffers.front();
        commandBuffers.pop();
        return cmd;
    }

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer;
    if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer.");
    }

    return cmdBuffer;
}

void CommandBufferManager::releaseCommandBuffer(VkCommandBuffer cmd) {
    std::lock_guard<std::mutex> lock(mutex);
    commandBuffers.push(cmd);
}
