#ifndef VULKAN_ERROR_H
#define VULKAN_ERROR_H

#include <stdexcept>
#include <string>

// Define VulkanOperationType enum
enum class VulkanOperationType {
    MatMul,
    Conv2D,
    ReLU,
    Sigmoid,
    Softmax,
    MaxPool,
    BatchNorm,
    Add
};

class VulkanError : public std::runtime_error {
public:
    explicit VulkanError(const std::string& message) : std::runtime_error(message) {}
};

#endif // VULKAN_ERROR_H