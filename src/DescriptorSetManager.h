#ifndef DESCRIPTORSETMANAGER_H
#define DESCRIPTORSETMANAGER_H

#include <vulkan/vulkan.h>
#include <vector>

class DescriptorSetManager {
public:
    DescriptorSetManager(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout);
    ~DescriptorSetManager();

    VkDescriptorSet allocateDescriptorSet();
    void updateDescriptorSet(VkDescriptorSet descriptorSet, const std::vector<VkBuffer>& inputBuffers, VkBuffer outputBuffer);

private:
    VkDevice device;
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayout;
};

#endif // DESCRIPTORSETMANAGER_H
