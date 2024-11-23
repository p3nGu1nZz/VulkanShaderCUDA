#include "VulkanOperations.h"
#include "vulkan_globals.h"
#include "VulkanContext.h"
#include "VulkanSync.h"
#include "DescriptorSetManager.h"
#include <spdlog/spdlog.h>

namespace vulkan_ops {

template<typename PushConstants>
void executeShader(
    VulkanOperationType opType,
    const VulkanTensor& inputA,
    const VulkanTensor* inputB,
    const VulkanTensor* inputC,
    VulkanTensor& output,
    const PushConstants* pushConstants
) {
    spdlog::debug("Executing shader for operation: {}", static_cast<int>(opType));

    // Get VulkanContext pointer
    VulkanContext* context = vulkan_globals::getContext();

    // Create pipeline key
    PipelineKey key = {
        opType,
        {}, // Input formats
        {}, // Output formats
        256, 1, 1 // Workgroup sizes
    };

    // Get pipeline and layout
    VkPipeline pipeline = context->getPipelineManager()->getPipeline(key);
    VkPipelineLayout pipelineLayout = context->getPipelineManager()->getPipelineLayout(key);

    // Acquire command buffer
    VkCommandBuffer commandBuffer = context->getCommandBufferManager()->acquireCommandBuffer();

    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkResult result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin command buffer");
    }

    // Bind pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    // Allocate and update descriptor set
    std::shared_ptr<DescriptorSetManager> descriptorManager = context->getDescriptorSetManager();
    VkDescriptorSet descriptorSet = descriptorManager->allocateDescriptorSet();
    
    std::vector<VkBuffer> buffers = { inputA.getBuffer() };
    if (inputB) buffers.push_back(inputB->getBuffer());
    if (inputC) buffers.push_back(inputC->getBuffer());
    descriptorManager->updateDescriptorSet(descriptorSet, buffers, output.getBuffer());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout,
        0,
        1,
        &descriptorSet,
        0,
        nullptr
    );

    // Push constants if provided
    if (pushConstants) {
        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(PushConstants),
            pushConstants
        );
    }

    // Calculate dispatch dimensions
    uint32_t globalWorkSizeX = (output.getSize() + 255) / 256;
    uint32_t globalWorkSizeY = 1;
    uint32_t globalWorkSizeZ = 1;

    // Dispatch compute shader
    vkCmdDispatch(commandBuffer, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ);

    // Memory barrier
    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    // End command buffer
    result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer");
    }

    // Submit command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    // Create fence for synchronization
    VulkanSync::ScopedGPUWait scopedWait(vulkan_globals::device);
    
    result = vkQueueSubmit(vulkan_globals::computeQueue, 1, &submitInfo, scopedWait.get());
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer");
    }

    // Wait for completion
    scopedWait.wait();

    // Release command buffer
    context->getCommandBufferManager()->releaseCommandBuffer(commandBuffer);
}

void executeAdd(const VulkanTensor& inputA, const VulkanTensor& inputB, VulkanTensor& output) {
    if (inputA.getSize() != inputB.getSize() || inputA.getSize() != output.getSize()) {
        throw std::runtime_error("Tensor size mismatch in Add operation");
    }

    AddPushConstants pushConstants = {
        static_cast<uint32_t>(inputA.getSize() / sizeof(float))
    };
    
    executeShader(VulkanOperationType::Add, inputA, &inputB, nullptr, output, &pushConstants);
}

void executeMatMul(const VulkanTensor& a, const VulkanTensor& b, VulkanTensor& c, 
                  uint32_t M, uint32_t K, uint32_t N) {
    // Validate dimensions
    if (a.getSize() != M * K * sizeof(float) ||
        b.getSize() != K * N * sizeof(float) ||
        c.getSize() != M * N * sizeof(float)) {
        throw std::runtime_error("Tensor dimensions mismatch in MatMul operation");
    }

    MatMulPushConstants pushConstants = {
        M, K, N
    };

    executeShader(VulkanOperationType::MatMul, a, &b, nullptr, c, &pushConstants);
}

void executeReLU(const VulkanTensor& input, VulkanTensor& output) {
    if (input.getSize() != output.getSize()) {
        throw std::runtime_error("Tensor size mismatch in ReLU operation");
    }

    ReLUPushConstants pushConstants = {
        static_cast<uint32_t>(input.getSize() / sizeof(float))
    };

    executeShader(VulkanOperationType::ReLU, input, nullptr, nullptr, output, &pushConstants);
}

void executeSigmoid(const VulkanTensor& input, VulkanTensor& output) {
    if (input.getSize() != output.getSize()) {
        throw std::runtime_error("Tensor size mismatch in Sigmoid operation");
    }

    SigmoidPushConstants pushConstants = {
        static_cast<uint32_t>(input.getSize() / sizeof(float))
    };

    executeShader(VulkanOperationType::Sigmoid, input, nullptr, nullptr, output, &pushConstants);
}

void executeSoftmax(const VulkanTensor& input, VulkanTensor& output) {
    if (input.getSize() != output.getSize()) {
        throw std::runtime_error("Tensor size mismatch in Softmax operation");
    }

    SoftmaxPushConstants pushConstants = {
        static_cast<uint32_t>(input.getSize() / sizeof(float))
    };

    executeShader(VulkanOperationType::Softmax, input, nullptr, nullptr, output, &pushConstants);
}

void executeConv2D(const VulkanTensor& input, const VulkanTensor& kernel,
                  VulkanTensor& output, const Conv2DPushConstants& pushConstants) {
    // Validate dimensions
    uint32_t expectedOutputWidth = 
        (pushConstants.input_width + 2 * pushConstants.padding - pushConstants.kernel_size) / 
        pushConstants.stride + 1;
    uint32_t expectedOutputHeight = 
        (pushConstants.input_height + 2 * pushConstants.padding - pushConstants.kernel_size) / 
        pushConstants.stride + 1;

    if (output.getWidth() != expectedOutputWidth || 
        output.getHeight() != expectedOutputHeight ||
        output.getChannels() != pushConstants.output_channels) {
        throw std::runtime_error("Output tensor dimensions mismatch in Conv2D operation");
    }

    size_t expectedKernelSize = 
        pushConstants.kernel_size * 
        pushConstants.kernel_size * 
        pushConstants.input_channels * 
        pushConstants.output_channels * 
        sizeof(float);

    if (kernel.getSize() != expectedKernelSize) {
        throw std::runtime_error("Kernel tensor dimensions mismatch in Conv2D operation");
    }

    executeShader(VulkanOperationType::Conv2D, input, &kernel, nullptr, output, &pushConstants);
}

void executeMaxPool(const VulkanTensor& input, VulkanTensor& output,
                   uint32_t width, uint32_t height, uint32_t channels,
                   uint32_t poolSizeX, uint32_t poolSizeY,
                   uint32_t strideX, uint32_t strideY) {
    // Validate dimensions
    uint32_t expectedOutputWidth = (width - poolSizeX) / strideX + 1;
    uint32_t expectedOutputHeight = (height - poolSizeY) / strideY + 1;

    if (output.getWidth() != expectedOutputWidth ||
        output.getHeight() != expectedOutputHeight ||
        output.getChannels() != channels) {
        throw std::runtime_error("Output tensor dimensions mismatch in MaxPool operation");
    }

    MaxPoolPushConstants pushConstants = {
        width, height, channels, 1,  // batch_size = 1
        poolSizeX, poolSizeY,
        strideX, strideY
    };

    executeShader(VulkanOperationType::MaxPool, input, nullptr, nullptr, output, &pushConstants);
}

void executeBatchNorm(const VulkanTensor& input, const VulkanTensor& gamma,
                     const VulkanTensor& beta, VulkanTensor& output,
                     uint32_t size, float epsilon) {
    // Validate dimensions
    if (input.getSize() != output.getSize() ||
        gamma.getSize() != size * sizeof(float) ||
        beta.getSize() != size * sizeof(float)) {
        throw std::runtime_error("Tensor dimensions mismatch in BatchNorm operation");
    }

    BatchNormPushConstants pushConstants = {
        size,
        epsilon
    };

    executeShader(VulkanOperationType::BatchNorm, input, &gamma, &beta, output, &pushConstants);
}

} // namespace vulkan_ops