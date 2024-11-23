#ifndef PIPELINE_MANAGER_H
#define PIPELINE_MANAGER_H

#include <vulkan/vulkan.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include "VulkanError.h"
#include "ShaderManager.h"
#include "PushConstants.h"

struct PipelineKey {
    VulkanOperationType opType;
    std::vector<std::string> inputFormats;
    std::vector<std::string> outputFormats;
    uint32_t workgroupSizeX;
    uint32_t workgroupSizeY;
    uint32_t workgroupSizeZ;

    bool operator==(const PipelineKey& other) const {
        return opType == other.opType &&
               inputFormats == other.inputFormats &&
               outputFormats == other.outputFormats &&
               workgroupSizeX == other.workgroupSizeX &&
               workgroupSizeY == other.workgroupSizeY &&
               workgroupSizeZ == other.workgroupSizeZ;
    }
};

struct PipelineKeyHash {
    std::size_t operator()(const PipelineKey& key) const;
};

class PipelineManager {
public:
    PipelineManager(std::shared_ptr<ShaderManager> shaderManager, VkDevice device);
    ~PipelineManager();

    VkPipeline getPipeline(const PipelineKey& key);
    VkPipelineLayout getPipelineLayout(const PipelineKey& key);

private:
    std::shared_ptr<ShaderManager> shaderManager;
    VkDevice device;
    std::unordered_map<PipelineKey, VkPipeline, PipelineKeyHash> pipelines;
    std::unordered_map<PipelineKey, VkPipelineLayout, PipelineKeyHash> pipelineLayouts;

    std::string getShaderName(VulkanOperationType opType) const;
    std::vector<char> loadShaderCode(const std::string& shaderName) const;
};

#endif // PIPELINE_MANAGER_H
