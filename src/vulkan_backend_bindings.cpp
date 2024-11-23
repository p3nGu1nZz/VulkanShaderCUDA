// src\vulkan_backend_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>

#include "VulkanOperations.h"
#include "VulkanContext.h"
#include "VulkanTensor.h"
#include "vulkan_globals.h"
#include "OnnxModelParser.h"

namespace py = pybind11;

PYBIND11_MODULE(vulkan_backend, m) {
    m.doc() = "Vulkan Backend for PyTorch Operations";

    // Initialize with proper shader path discovery
    m.def("init_vulkan", []() -> bool {
        try {
            // Get the path to the current module
            auto module = py::module::import("__main__");
            auto module_path = std::filesystem::path(module.attr("__file__").cast<std::string>());
            
            // Initialize shader directory based on module location
            vulkan_globals::setShaderDirectory(module_path);
            
            bool success = vulkan_globals::initializeVulkan();
            if (success && vulkan_globals::getContext()) {
                vulkan_globals::device = vulkan_globals::getContext()->getDevice();
                return true;
            }
            return false;
        }
        catch (const std::exception& e) {
            spdlog::error("Failed to initialize Vulkan: {}", e.what());
            return false;
        }
    }, "Initialize Vulkan context");

    m.def("cleanup_vulkan", []() -> void {
        vulkan_globals::cleanupVulkan();
        vulkan_globals::device = VK_NULL_HANDLE;
    }, "Cleanup Vulkan context");

    m.def("is_vulkan_initialized", []() -> bool {
        return vulkan_globals::getContext() != nullptr && vulkan_globals::device != VK_NULL_HANDLE;
    }, "Check if Vulkan context is initialized");

    // Import ONNX model
    m.def("import_onnx_model", [](const std::string& modelPath) -> void {
        try {
            OnnxModelParser parser(modelPath);
            spdlog::info("ONNX model imported successfully: {}", modelPath);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Failed to import ONNX model: " + std::string(e.what()));
        }
    }, "Import an ONNX model for execution on Vulkan backend");

    // Add Operation
    m.def("vulkan_add", [](py::array_t<float> inputA, py::array_t<float> inputB, py::array_t<float> output) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input arrays are contiguous
            if (!(inputA.flags() & py::array::c_style)) {
                throw std::runtime_error("Input A must be a contiguous array.");
            }
            if (!(inputB.flags() & py::array::c_style)) {
                throw std::runtime_error("Input B must be a contiguous array.");
            }
            if (!(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Output must be a contiguous array.");
            }

            auto buf_a = inputA.request();
            auto buf_b = inputB.request();
            auto buf_c = output.request();

            // Ensure all arrays have the same size
            if (buf_a.size != buf_b.size || buf_a.size != buf_c.size) {
                throw std::runtime_error("Input and output arrays must have the same size.");
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorA(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_a.size * sizeof(float)),
                1, 1, 1, 1,
                buf_a.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorB(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_b.size * sizeof(float)),
                1, 1, 1, 1,
                buf_b.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorC(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_c.size * sizeof(float)),
                1, 1, 1, 1,
                nullptr,
                TensorLayout::Layout::LINEAR
            );

            // Execute Add operation
            vulkan_ops::executeAdd(tensorA, tensorB, tensorC);

            // Download result
            tensorC.download(buf_c.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Add operation failed: " + std::string(e.what()));
        }
    }, "Execute addition operation on Vulkan backend");

    // Matrix Multiplication operation
    m.def("vulkan_matmul", [](py::array_t<float> inputA, py::array_t<float> inputB, py::array_t<float> output,
                            uint32_t M, uint32_t K, uint32_t N) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input arrays are contiguous
            if (!(inputA.flags() & py::array::c_style)) {
                throw std::runtime_error("Input A must be a contiguous array.");
            }
            if (!(inputB.flags() & py::array::c_style)) {
                throw std::runtime_error("Input B must be a contiguous array.");
            }
            if (!(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Output must be a contiguous array.");
            }

            auto buf_a = inputA.request();
            auto buf_b = inputB.request();
            auto buf_c = output.request();

            // Calculate expected sizes
            size_t expected_size_a = static_cast<size_t>(M * K);
            size_t expected_size_b = static_cast<size_t>(K * N);
            size_t expected_size_c = static_cast<size_t>(M * N);

            if (buf_a.size != expected_size_a) {
                throw std::runtime_error("Input A size mismatch. Expected: " + std::to_string(expected_size_a) + 
                                       ", Got: " + std::to_string(buf_a.size));
            }
            if (buf_b.size != expected_size_b) {
                throw std::runtime_error("Input B size mismatch. Expected: " + std::to_string(expected_size_b) + 
                                       ", Got: " + std::to_string(buf_b.size));
            }
            if (buf_c.size != expected_size_c) {
                throw std::runtime_error("Output size mismatch. Expected: " + std::to_string(expected_size_c) + 
                                       ", Got: " + std::to_string(buf_c.size));
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorA(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_a.size * sizeof(float)),
                K, M, 1, 1,
                buf_a.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorB(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_b.size * sizeof(float)),
                N, K, 1, 1,
                buf_b.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorC(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_c.size * sizeof(float)),
                N, M, 1, 1,
                nullptr,
                TensorLayout::Layout::LINEAR
            );

            // Execute MatMul operation
            vulkan_ops::executeMatMul(tensorA, tensorB, tensorC, M, K, N);

            // Download result
            tensorC.download(buf_c.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Matrix multiplication failed: " + std::string(e.what()));
        }
    }, "Execute matrix multiplication on Vulkan backend");

    // ReLU operation
    m.def("vulkan_relu", [](py::array_t<float> input, py::array_t<float> output) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input arrays are contiguous
            if (!(input.flags() & py::array::c_style)) {
                throw std::runtime_error("Input must be a contiguous array.");
            }
            if (!(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Output must be a contiguous array.");
            }

            auto buf_input = input.request();
            auto buf_output = output.request();

            // Ensure sizes match
            if (buf_input.size != buf_output.size) {
                throw std::runtime_error("Input and output arrays must have the same size.");
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorInput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_input.size * sizeof(float)),
                1, 1, 1, 1,
                buf_input.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                1, 1, 1, 1,
                nullptr,
                TensorLayout::Layout::LINEAR
            );

            // Execute ReLU operation
            vulkan_ops::executeReLU(tensorInput, tensorOutput);

            // Download result
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("ReLU operation failed: " + std::string(e.what()));
        }
    }, "Execute ReLU activation on Vulkan backend");

    // Sigmoid operation
    m.def("vulkan_sigmoid", [](py::array_t<float> input, py::array_t<float> output) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input arrays are contiguous
            if (!(input.flags() & py::array::c_style)) {
                throw std::runtime_error("Input must be a contiguous array.");
            }
            if (!(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Output must be a contiguous array.");
            }

            auto buf_input = input.request();
            auto buf_output = output.request();

            // Ensure sizes match
            if (buf_input.size != buf_output.size) {
                throw std::runtime_error("Input and output arrays must have the same size.");
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorInput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_input.size * sizeof(float)),
                1, 1, 1, 1,
                buf_input.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                1, 1, 1, 1,
                nullptr,
                TensorLayout::Layout::LINEAR
            );

            // Execute Sigmoid operation
            vulkan_ops::executeSigmoid(tensorInput, tensorOutput);

            // Download result
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Sigmoid operation failed: " + std::string(e.what()));
        }
    }, "Execute Sigmoid activation on Vulkan backend");

    // Softmax operation
    m.def("vulkan_softmax", [](py::array_t<float> input, py::array_t<float> output) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input arrays are contiguous
            if (!(input.flags() & py::array::c_style)) {
                throw std::runtime_error("Input must be a contiguous array.");
            }
            if (!(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Output must be a contiguous array.");
            }

            auto buf_input = input.request();
            auto buf_output = output.request();

            // Ensure sizes match
            if (buf_input.size != buf_output.size) {
                throw std::runtime_error("Input and output arrays must have the same size.");
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorInput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_input.size * sizeof(float)),
                1, 1, 1, 1,
                buf_input.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                1, 1, 1, 1,
                nullptr,
                TensorLayout::Layout::LINEAR
            );

            // Execute Softmax operation
            vulkan_ops::executeSoftmax(tensorInput, tensorOutput);

            // Download result
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Softmax operation failed: " + std::string(e.what()));
        }
    }, "Execute Softmax operation on Vulkan backend");

    // Conv2D operation
    m.def("vulkan_conv2d", [](
        py::array_t<float> input,
        py::array_t<float> kernel,
        py::array_t<float> output,
        uint32_t input_width,
        uint32_t input_height, 
        uint32_t input_channels,
        uint32_t output_channels,
        uint32_t kernel_size,
        uint32_t padding,
        uint32_t stride) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input arrays are contiguous
            if (!(input.flags() & py::array::c_style)) {
                throw std::runtime_error("Input must be a contiguous array.");
            }
            if (!(kernel.flags() & py::array::c_style)) {
                throw std::runtime_error("Kernel must be a contiguous array.");
            }
            if (!(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Output must be a contiguous array.");
            }

            auto buf_input = input.request();
            auto buf_kernel = kernel.request();
            auto buf_output = output.request();

            // Calculate output dimensions
            uint32_t output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
            uint32_t output_height = (input_height + 2 * padding - kernel_size) / stride + 1;

            // Calculate expected sizes
            size_t expected_input_elements = static_cast<size_t>(input_width) * 
                                           static_cast<size_t>(input_height) * 
                                           static_cast<size_t>(input_channels);
            size_t expected_kernel_elements = static_cast<size_t>(kernel_size) * 
                                            static_cast<size_t>(kernel_size) * 
                                            static_cast<size_t>(input_channels) * 
                                            static_cast<size_t>(output_channels);
            size_t expected_output_elements = static_cast<size_t>(output_width) * 
                                            static_cast<size_t>(output_height) * 
                                            static_cast<size_t>(output_channels);

            // Validate sizes
            if (buf_input.size != expected_input_elements) {
                throw std::runtime_error(
                    "Input size mismatch. Expected: " + std::to_string(expected_input_elements) +
                    ", Got: " + std::to_string(buf_input.size)
                );
            }
            if (buf_kernel.size != expected_kernel_elements) {
                throw std::runtime_error(
                    "Kernel size mismatch. Expected: " + std::to_string(expected_kernel_elements) +
                    ", Got: " + std::to_string(buf_kernel.size)
                );
            }
            if (buf_output.size != expected_output_elements) {
                throw std::runtime_error(
                    "Output size mismatch. Expected: " + std::to_string(expected_output_elements) +
                    ", Got: " + std::to_string(buf_output.size)
                );
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorInput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_input.size * sizeof(float)),
                input_width,
                input_height,
                input_channels,
                1,
                buf_input.ptr,
                TensorLayout::Layout::NHWC
            );
            VulkanTensor tensorKernel(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_kernel.size * sizeof(float)),
                kernel_size,
                kernel_size,
                input_channels,
                output_channels,
                buf_kernel.ptr,
                TensorLayout::Layout::NHWC
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                output_width,
                output_height,
                output_channels,
                1,
                nullptr,
                TensorLayout::Layout::NHWC
            );

            // Set push constants
            Conv2DPushConstants pushConstants{
                input_width,
                input_height,
                input_channels,
                output_channels,
                kernel_size,
                padding,
                stride
            };

            // Execute convolution
            vulkan_ops::executeConv2D(tensorInput, tensorKernel, tensorOutput, pushConstants);

            // Download results
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Conv2D operation failed: " + std::string(e.what()));
        }
    }, 
    py::arg("input"), 
    py::arg("kernel"),
    py::arg("output"),
    py::arg("input_width"),
    py::arg("input_height"),
    py::arg("input_channels"),
    py::arg("output_channels"),
    py::arg("kernel_size"),
    py::arg("padding"),
    py::arg("stride")
    );

    // MaxPool operation
    m.def("vulkan_maxpool", [](py::array_t<float> input, py::array_t<float> output,
                              uint32_t width, uint32_t height, uint32_t channels,
                              uint32_t poolSizeX, uint32_t poolSizeY,
                              uint32_t strideX, uint32_t strideY) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input arrays are contiguous
            if (!(input.flags() & py::array::c_style)) {
                throw std::runtime_error("Input must be a contiguous array.");
            }
            if (!(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Output must be a contiguous array.");
            }

            auto buf_input = input.request();
            auto buf_output = output.request();

            uint32_t expected_output_width = (width - poolSizeX) / strideX + 1;
            uint32_t expected_output_height = (height - poolSizeY) / strideY + 1;
            uint32_t expected_output_size = expected_output_width * expected_output_height * channels;

            if (buf_input.size != width * height * channels) {
                throw std::runtime_error(
                    "Input dimensions mismatch. Expected size: " + std::to_string(width * height * channels) +
                    ", Got: " + std::to_string(buf_input.size)
                );
            }

            if (buf_output.size != expected_output_size) {
                throw std::runtime_error(
                    "Output dimensions mismatch. Expected size: " + std::to_string(expected_output_size) +
                    ", Got: " + std::to_string(buf_output.size)
                );
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorInput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_input.size * sizeof(float)),
                width,
                height,
                channels,
                1,
                buf_input.ptr,
                TensorLayout::Layout::NHWC
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                expected_output_width,
                expected_output_height,
                channels,
                1,
                nullptr,
                TensorLayout::Layout::NHWC
            );

            // Execute MaxPool operation
            vulkan_ops::executeMaxPool(tensorInput, tensorOutput, width, height, channels,
                                     poolSizeX, poolSizeY, strideX, strideY);

            // Download result
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("MaxPool operation failed: " + std::string(e.what()));
        }
    }, "Execute MaxPool2D operation on Vulkan backend");

    // BatchNorm operation
    m.def("vulkan_batchnorm", [](
        py::array_t<float> input,
        py::array_t<float> gamma,
        py::array_t<float> beta,
        py::array_t<float> output,
        uint32_t size,
        float epsilon) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input arrays are contiguous
            if (!(input.flags() & py::array::c_style)) {
                throw std::runtime_error("Input must be a contiguous array.");
            }
            if (!(gamma.flags() & py::array::c_style)) {
                throw std::runtime_error("Gamma must be a contiguous array.");
            }
            if (!(beta.flags() & py::array::c_style)) {
                throw std::runtime_error("Beta must be a contiguous array.");
            }
            if (!(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Output must be a contiguous array.");
            }

            auto buf_input = input.request();
            auto buf_gamma = gamma.request();
            auto buf_beta = beta.request();
            auto buf_output = output.request();

            if (buf_input.size != size || 
                buf_gamma.size != size ||
                buf_beta.size != size || 
                buf_output.size != size) {
                throw std::runtime_error(
                    "Size mismatch. All input, gamma, beta, and output tensors must have size: " + 
                    std::to_string(size)
                );
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorInput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_input.size * sizeof(float)),
                1, 1, size, 1,
                buf_input.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorGamma(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_gamma.size * sizeof(float)),
                1, 1, size, 1,
                buf_gamma.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorBeta(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_beta.size * sizeof(float)),
                1, 1, size, 1,
                buf_beta.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                1, 1, size, 1,
                nullptr,
                TensorLayout::Layout::LINEAR
            );

            // Execute BatchNorm operation
            vulkan_ops::executeBatchNorm(tensorInput, tensorGamma, tensorBeta, tensorOutput, size, epsilon);

            // Download result
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("BatchNorm operation failed: " + std::string(e.what()));
        }
    }, "Execute BatchNorm operation on Vulkan backend");

    // Model operations
    m.def("save_model", [](const std::string& filename) -> void {
        throw std::runtime_error("Model saving not yet implemented.");
    }, "Save the current model to a file");

    m.def("load_model", [](const std::string& filename) -> void {
        throw std::runtime_error("Model loading not yet implemented.");
    }, "Load a model from a file");

    m.def("initialize_distributed", [](uint32_t num_gpus) -> void {
        throw std::runtime_error("Distributed training not yet implemented.");
    }, "Initialize distributed training");

    m.def("enable_gradient_checkpointing", []() -> void {
        throw std::runtime_error("Gradient checkpointing not yet implemented.");
    }, "Enable gradient checkpointing");
}
