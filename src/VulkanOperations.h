#ifndef VULKAN_OPERATIONS_H
#define VULKAN_OPERATIONS_H

#include "VulkanTensor.h"
#include "PushConstants.h"

// Forward declarations
class VulkanTensor;

namespace vulkan_ops {
    // Execution functions declarations
    void executeAdd(const VulkanTensor& inputA, const VulkanTensor& inputB, VulkanTensor& output);
    
    void executeMatMul(const VulkanTensor& a, const VulkanTensor& b, VulkanTensor& c, 
                      uint32_t M, uint32_t K, uint32_t N);
    
    void executeReLU(const VulkanTensor& input, VulkanTensor& output);
    
    void executeSigmoid(const VulkanTensor& input, VulkanTensor& output);
    
    void executeSoftmax(const VulkanTensor& input, VulkanTensor& output);
    
    void executeConv2D(const VulkanTensor& input, const VulkanTensor& kernel, 
                      VulkanTensor& output, const Conv2DPushConstants& pushConstants);
    
    void executeMaxPool(const VulkanTensor& input, VulkanTensor& output, 
                       uint32_t width, uint32_t height, uint32_t channels,
                       uint32_t poolSizeX, uint32_t poolSizeY, 
                       uint32_t strideX, uint32_t strideY);
    
    void executeBatchNorm(const VulkanTensor& input, const VulkanTensor& gamma, 
                         const VulkanTensor& beta, VulkanTensor& output, 
                         uint32_t size, float epsilon);

    // Generic shader execution template
    template<typename PushConstants>
    void executeShader(VulkanOperationType opType,
                      const VulkanTensor& inputA,
                      const VulkanTensor* inputB,
                      const VulkanTensor* inputC,
                      VulkanTensor& output,
                      const PushConstants* pushConstants);
}

#endif // VULKAN_OPERATIONS_H