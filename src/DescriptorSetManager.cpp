// src\DescriptorSetManager.cpp
#include "DescriptorSetManager.h"
#include "spdlog/spdlog.h"

DescriptorSetManager::DescriptorSetManager(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout)
    : device(device), descriptorPool(descriptorPool), descriptorSetLayout(descriptorSetLayout) {
    spdlog::info("DescriptorSetManager created.");
}

DescriptorSetManager::~DescriptorSetManager() {
    spdlog::info("DescriptorSetManager destroyed.");
}

VkDescriptorSet DescriptorSetManager::allocateDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set.");
    }

    return descriptorSet;
}

void DescriptorSetManager::updateDescriptorSet(VkDescriptorSet descriptorSet, const std::vector<VkBuffer>& inputBuffers, VkBuffer outputBuffer) {
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkWriteDescriptorSet> descriptorWrites;

    // Reserve space to avoid reallocations
    bufferInfos.reserve(inputBuffers.size() + 1);
    descriptorWrites.reserve(inputBuffers.size() + 1);

    // Process input buffers
    for (size_t i = 0; i < inputBuffers.size(); i++) {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = inputBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = VK_WHOLE_SIZE;
        bufferInfos.push_back(bufferInfo);

        VkWriteDescriptorSet writeDescriptor = {};
        writeDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptor.dstSet = descriptorSet;
        writeDescriptor.dstBinding = static_cast<uint32_t>(i); // Binding index for input buffers
        writeDescriptor.dstArrayElement = 0;
        writeDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptor.descriptorCount = 1;
        writeDescriptor.pBufferInfo = &bufferInfos.back();
        descriptorWrites.push_back(writeDescriptor);
    }

    // Add output buffer
    if (outputBuffer != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo outputBufferInfo = {};
        outputBufferInfo.buffer = outputBuffer;
        outputBufferInfo.offset = 0;
        outputBufferInfo.range = VK_WHOLE_SIZE;
        bufferInfos.push_back(outputBufferInfo);

        VkWriteDescriptorSet writeDescriptor = {};
        writeDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptor.dstSet = descriptorSet;
        writeDescriptor.dstBinding = static_cast<uint32_t>(inputBuffers.size()); // Binding index for output buffer
        writeDescriptor.dstArrayElement = 0;
        writeDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptor.descriptorCount = 1;
        writeDescriptor.pBufferInfo = &bufferInfos.back();
        descriptorWrites.push_back(writeDescriptor);
    }

    // Update all descriptors at once
    vkUpdateDescriptorSets(
        device,
        static_cast<uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        nullptr
    );
}
