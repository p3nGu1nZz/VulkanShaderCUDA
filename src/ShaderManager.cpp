// src\ShaderManager.cpp
#include "ShaderManager.h"
#include <spdlog/spdlog.h>

ShaderManager::ShaderManager(VkDevice device)
    : device(device) {}

ShaderManager::~ShaderManager() {
    for (auto& [name, shaderModule] : shaderModules) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        spdlog::info("Shader module destroyed: {}", name);
    }
}

VkShaderModule ShaderManager::getShaderModule(const std::string& shaderName, const std::vector<char>& code) {
    auto it = shaderModules.find(shaderName);
    if (it != shaderModules.end()) {
        return it->second;
    }

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();

    // Ensure code is aligned to 4 bytes as required by Vulkan
    if (code.size() % 4 != 0) {
        throw VulkanError("Shader code size is not a multiple of 4 bytes.");
    }
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to create shader module: " + shaderName);
    }

    // Optional: Verify shader compilation by creating a dummy pipeline
    // This step is skipped here but recommended during development using validation layers

    shaderModules[shaderName] = shaderModule;
    spdlog::info("Shader module created: {}", shaderName);
    return shaderModule;
}
