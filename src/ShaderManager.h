#ifndef SHADER_MANAGER_H
#define SHADER_MANAGER_H

#include <vulkan/vulkan.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include "VulkanError.h"

class ShaderManager {
public:
    ShaderManager(VkDevice device);
    ~ShaderManager();

    VkShaderModule getShaderModule(const std::string& shaderName, const std::vector<char>& code);

private:
    VkDevice device;
    std::unordered_map<std::string, VkShaderModule> shaderModules;
};

#endif // SHADER_MANAGER_H