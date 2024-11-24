// src\CommandBufferManager.cpp
#include "CommandBufferManager.h"
#include "spdlog/spdlog.h"

// Constructor
CommandBufferManager::CommandBufferManager(VkDevice device, VkCommandPool commandPool)
    : device(device), commandPool(commandPool) {
    spdlog::info("CommandBufferManager created.");
}

// Destructor: Cleans up all command buffers in the pool
CommandBufferManager::~CommandBufferManager() {
    std::lock_guard<std::mutex> lock(mutex);

    while (!commandBuffers.empty()) {
        VkCommandBuffer cmd = commandBuffers.front();
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
        commandBuffers.pop();
    }
    spdlog::info("CommandBufferManager destroyed.");
}

// Acquire a command buffer from the pool or allocate a new one if the pool is empty
VkCommandBuffer CommandBufferManager::acquireCommandBuffer() {
    std::lock_guard<std::mutex> lock(mutex);

    if (!commandBuffers.empty()) {
        VkCommandBuffer cmd = commandBuffers.front();
        commandBuffers.pop();
        spdlog::debug("Acquired command buffer from pool.");
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

    spdlog::debug("Allocated new command buffer.");
    return cmdBuffer;
}

// Release a command buffer back to the pool for reuse
void CommandBufferManager::releaseCommandBuffer(VkCommandBuffer cmd) {
    std::lock_guard<std::mutex> lock(mutex);
    commandBuffers.push(cmd);
    spdlog::debug("Released command buffer back to pool.");
}
