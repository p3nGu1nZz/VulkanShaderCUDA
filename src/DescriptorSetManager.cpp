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
    std::vector<VkDescriptorBufferInfo> bufferInfos(inputBuffers.size());

    for (size_t i = 0; i < inputBuffers.size(); i++) {
        bufferInfos[i].buffer = inputBuffers[i];
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = VK_WHOLE_SIZE;
    }

    VkDescriptorBufferInfo outputBufferInfo = {};
    outputBufferInfo.buffer = outputBuffer;
    outputBufferInfo.offset = 0;
    outputBufferInfo.range = VK_WHOLE_SIZE;

    std::vector<VkWriteDescriptorSet> descriptorWrites(bufferInfos.size() + 1);

    for (size_t i = 0; i < bufferInfos.size(); i++) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = static_cast<uint32_t>(i);
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }

    descriptorWrites.back().sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites.back().dstSet = descriptorSet;
    descriptorWrites.back().dstBinding = static_cast<uint32_t>(bufferInfos.size());
    descriptorWrites.back().dstArrayElement = 0;
    descriptorWrites.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites.back().descriptorCount = 1;
    descriptorWrites.back().pBufferInfo = &outputBufferInfo;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}
