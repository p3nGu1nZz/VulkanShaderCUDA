// PipelineManager.cpp

#include "PipelineManager.h"
#include "vulkan_globals.h"
#include "spdlog/spdlog.h"
#include <fstream>
#include <filesystem>

PipelineManager::PipelineManager(std::shared_ptr<ShaderManager> shaderManager, VkDevice device)
    : shaderManager(shaderManager), device(device) {}

PipelineManager::~PipelineManager() {
    // Clean up pipelines
    for (auto& [key, pipeline] : pipelines) {
        vkDestroyPipeline(device, pipeline, nullptr);
    }

    // Clean up pipeline layouts
    for (auto& [key, layout] : pipelineLayouts) {
        vkDestroyPipelineLayout(device, layout, nullptr);
    }
}

std::size_t PipelineKeyHash::operator()(const PipelineKey& key) const {
    std::size_t res = std::hash<int>()(static_cast<int>(key.opType));
    
    // Combine hash of input formats
    for (const auto& format : key.inputFormats) {
        res ^= std::hash<std::string>()(format) + 0x9e3779b9 + (res << 6) + (res >> 2);
    }

    // Combine hash of output formats
    for (const auto& format : key.outputFormats) {
        res ^= std::hash<std::string>()(format) + 0x9e3779b9 + (res << 6) + (res >> 2);
    }

    // Combine workgroup size hashes
    res ^= std::hash<uint32_t>()(key.workgroupSizeX) + 0x9e3779b9 + (res << 6) + (res >> 2);
    res ^= std::hash<uint32_t>()(key.workgroupSizeY) + 0x9e3779b9 + (res << 6) + (res >> 2);
    res ^= std::hash<uint32_t>()(key.workgroupSizeZ) + 0x9e3779b9 + (res << 6) + (res >> 2);

    return res;
}

std::string PipelineManager::getShaderName(VulkanOperationType opType) const {
    switch (opType) {
        case VulkanOperationType::MatMul:   return "matmul.comp.spv";
        case VulkanOperationType::Conv2D:   return "conv2d.comp.spv";
        case VulkanOperationType::ReLU:     return "relu.comp.spv";
        case VulkanOperationType::Sigmoid:  return "sigmoid.comp.spv";
        case VulkanOperationType::Softmax:  return "softmax.comp.spv";
        case VulkanOperationType::MaxPool:  return "pooling.comp.spv";
        case VulkanOperationType::BatchNorm: return "batchnorm.comp.spv";
        case VulkanOperationType::Add:      return "add.comp.spv";
        default:
            throw VulkanError("Unknown Vulkan operation type.");
    }
}

std::vector<char> PipelineManager::loadShaderCode(const std::string& shaderName) const {
    std::filesystem::path shaderPath = vulkan_globals::shader_directory / shaderName;
    
    // Convert path to string for opening file
    std::string shaderPathStr = shaderPath.string();
    std::ifstream file(shaderPathStr, std::ios::binary | std::ios::ate);
    
    if (!file.is_open()) {
        throw VulkanError("Failed to open shader file: " + shaderPathStr);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    spdlog::info("Successfully loaded shader: {}", shaderPathStr);
    return buffer;
}

VkPipeline PipelineManager::getPipeline(const PipelineKey& key) {
    // Check if pipeline already exists
    auto it = pipelines.find(key);
    if (it != pipelines.end()) {
        return it->second;
    }

    // Create new pipeline layout
    VkPipelineLayout pipelineLayout = getPipelineLayout(key);

    // Load shader module
    std::string shaderName = getShaderName(key.opType);
    std::vector<char> shaderCode = loadShaderCode(shaderName);
    VkShaderModule shaderModule = shaderManager->getShaderModule(shaderName, shaderCode);

    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";

    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to create compute pipeline.");
    }

    // Store and return the created pipeline
    pipelines[key] = pipeline;
    return pipeline;
}

VkPipelineLayout PipelineManager::getPipelineLayout(const PipelineKey& key) {
    // Check if pipeline layout already exists
    auto it = pipelineLayouts.find(key);
    if (it != pipelineLayouts.end()) {
        return it->second;
    }

    // Define push constants based on the operation type
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    switch (key.opType) {
        case VulkanOperationType::MatMul:
            pushConstantRange.size = sizeof(MatMulPushConstants);
            break;
        case VulkanOperationType::Conv2D:
            pushConstantRange.size = sizeof(Conv2DPushConstants);
            break;
        case VulkanOperationType::ReLU:
            pushConstantRange.size = sizeof(ReLUPushConstants);
            break;
        case VulkanOperationType::Sigmoid:
            pushConstantRange.size = sizeof(SigmoidPushConstants);
            break;
        case VulkanOperationType::Softmax:
            pushConstantRange.size = sizeof(SoftmaxPushConstants);
            break;
        case VulkanOperationType::MaxPool:
            pushConstantRange.size = sizeof(MaxPoolPushConstants);
            break;
        case VulkanOperationType::BatchNorm:
            pushConstantRange.size = sizeof(BatchNormPushConstants);
            break;
        case VulkanOperationType::Add:
            pushConstantRange.size = sizeof(AddPushConstants);
            break;
        default:
            pushConstantRange.size = 0;
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0; // No descriptor sets used in this example
    pipelineLayoutInfo.pSetLayouts = nullptr;
    if (pushConstantRange.size > 0) {
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    }

    VkPipelineLayout pipelineLayout;
    VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
    if (result != VK_SUCCESS) {
        throw VulkanError("Failed to create pipeline layout.");
    }

    // Store and return the created layout
    pipelineLayouts[key] = pipelineLayout;
    return pipelineLayout;
}