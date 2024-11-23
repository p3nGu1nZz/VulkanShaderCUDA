// src\vulkan_backend_bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <cstdint> // For int64_t

// Include your Vulkan backend headers
#include "VulkanOperations.h"    // Contains executeAdd, executeMatMul, etc.
#include "VulkanContext.h"       // Manages Vulkan context
#include "VulkanTensor.h"        // VulkanTensor class definition
#include "vulkan_globals.h"      // Global Vulkan context management
#include "OnnxModelParser.h"     // ONNX model parser
#include "Utils.h"               // For checkSize function and utility macros

namespace py = pybind11;

// -----------------------------
// Binding TensorLayout::Layout Enum
// -----------------------------
PYBIND11_MODULE(vulkan_backend, m) {
    m.doc() = "Vulkan Backend for PyTorch Operations";

    // Bind the TensorLayout::Layout enum to Python
    py::enum_<TensorLayout::Layout>(m, "Layout")
        .value("NHWC", TensorLayout::Layout::NHWC, "Batch, Height, Width, Channels")
        .value("NCHW", TensorLayout::Layout::NCHW, "Batch, Channels, Height, Width")
        .value("LINEAR", TensorLayout::Layout::LINEAR, "Flat layout")
        .export_values();

    // -----------------------------
    // Vulkan Initialization
    // -----------------------------
    m.def("init_vulkan", []() -> bool {
        try {
            // Get the path to the current module
            auto main_module = py::module::import("__main__");
            std::string main_file = main_module.attr("__file__").cast<std::string>();
            std::filesystem::path module_path = std::filesystem::path(main_file).parent_path();

            // Initialize shader directory based on module location
            vulkan_globals::setShaderDirectory(module_path);

            // Initialize Vulkan context
            bool success = vulkan_globals::initializeVulkan();
            if (success && vulkan_globals::getContext()) {
                vulkan_globals::device = vulkan_globals::getContext()->getDevice();
                spdlog::info("Vulkan initialized successfully.");
                return true;
            }
            spdlog::error("Vulkan initialization failed.");
            return false;
        }
        catch (const std::exception& e) {
            spdlog::error("Failed to initialize Vulkan: {}", e.what());
            return false;
        }
    }, "Initialize Vulkan context");

    // -----------------------------
    // Vulkan Cleanup
    // -----------------------------
    m.def("cleanup_vulkan", []() -> void {
        try {
            vulkan_globals::cleanupVulkan();
            vulkan_globals::device = VK_NULL_HANDLE;
            spdlog::info("Vulkan cleaned up successfully.");
        }
        catch (const std::exception& e) {
            spdlog::error("Failed to cleanup Vulkan: {}", e.what());
            throw std::runtime_error("Vulkan cleanup failed.");
        }
    }, "Cleanup Vulkan context");

    // -----------------------------
    // Check Vulkan Initialization
    // -----------------------------
    m.def("is_vulkan_initialized", []() -> bool {
        return vulkan_globals::getContext() != nullptr && vulkan_globals::device != VK_NULL_HANDLE;
    }, "Check if Vulkan context is initialized");

    // -----------------------------
    // Import ONNX Model
    // -----------------------------
    m.def("import_onnx_model", [](const std::string& modelPath) -> void {
        try {
            OnnxModelParser parser(modelPath);
            spdlog::info("ONNX model imported successfully: {}", modelPath);
        }
        catch (const std::exception& e) {
            spdlog::error("Failed to import ONNX model: {}", e.what());
            throw std::runtime_error("ONNX model import failed.");
        }
    }, "Import an ONNX model for execution on Vulkan backend");

    // -----------------------------
    // VulkanTensor Python Class
    // -----------------------------
    py::class_<VulkanTensor>(m, "VulkanTensor")
        .def(py::init<>(), "Default constructor for VulkanTensor.")
        .def_property("data", 
            // Getter: Download data from Vulkan buffer to NumPy array
            [](VulkanTensor &self) -> py::array_t<float> {
                // Get tensor dimensions and layout
                uint32_t n = self.getN();
                uint32_t c = self.getC();
                uint32_t h = self.getH();
                uint32_t w = self.getW();
                TensorLayout::Layout layout = self.getLayout();

                // Define shape based on layout
                std::vector<int64_t> shape;
                if (layout == TensorLayout::Layout::NHWC) {
                    shape = { static_cast<int64_t>(n), 
                              static_cast<int64_t>(h), 
                              static_cast<int64_t>(w), 
                              static_cast<int64_t>(c) };
                }
                else if (layout == TensorLayout::Layout::NCHW) {
                    shape = { static_cast<int64_t>(n), 
                              static_cast<int64_t>(c), 
                              static_cast<int64_t>(h), 
                              static_cast<int64_t>(w) };
                }
                else { // LINEAR
                    shape = { static_cast<int64_t>(n * c * h * w) };
                }

                // Allocate NumPy array without initializing data
                py::array_t<float> result(shape, nullptr);

                // Get pointer to NumPy data
                float* ptr = static_cast<float*>(result.request().ptr);

                // Download data from Vulkan buffer to host
                self.download(ptr);

                return result;
            },
            // Setter: Upload data from NumPy array to Vulkan buffer
            [](VulkanTensor &self, py::array_t<float> input) {
                // Ensure input array is contiguous
                if (!(input.flags() & py::array::c_style)) {
                    throw std::runtime_error("Input array must be contiguous.");
                }

                // Calculate expected size based on tensor dimensions
                size_t expected_size = static_cast<size_t>(self.getN()) *
                                       static_cast<size_t>(self.getC()) *
                                       static_cast<size_t>(self.getH()) *
                                       static_cast<size_t>(self.getW());

                if (input.size() != expected_size) {
                    throw std::runtime_error("Input array size does not match VulkanTensor size.");
                }

                // Get pointer to input data
                const float* ptr = static_cast<const float*>(input.request().ptr);

                // Upload data from host to Vulkan buffer
                self.upload(ptr);
            },
            "Data buffer of the tensor."
        )
        .def("get_shape", [](const VulkanTensor& self) -> std::vector<int64_t> {
            return { static_cast<int64_t>(self.getN()),
                     static_cast<int64_t>(self.getC()),
                     static_cast<int64_t>(self.getH()),
                     static_cast<int64_t>(self.getW()) };
        }, "Get the shape of the tensor.")
        .def("get_layout", &VulkanTensor::getLayout, "Get the tensor layout.")
        .def("set_layout", &VulkanTensor::setLayout, "Set the tensor layout.");

    // -----------------------------
    // Factory Function to Create VulkanTensor
    // -----------------------------
    m.def("create_vulkan_tensor", [](py::array_t<float> data, 
                                     uint32_t n, 
                                     uint32_t h, 
                                     uint32_t w, 
                                     uint32_t c, 
                                     TensorLayout::Layout layout) -> VulkanTensor {
        if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
            throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
        }

        if (!(data.flags() & py::array::c_style)) {
            throw std::runtime_error("Input array must be contiguous.");
        }

        size_t expected_size = static_cast<size_t>(n) * h * w * c;
        if (data.size() != expected_size) {
            throw std::runtime_error("Input array size does not match specified dimensions.");
        }

        auto buf = data.request();
        const float* ptr = static_cast<const float*>(buf.ptr);

        auto context = vulkan_globals::getContext();
        VulkanTensor tensor(
            context->getMemoryManager(),
            context->getBufferPool(),
            checkSize(buf.size * sizeof(float)),
            w, h, c, n, // w, h, c, n
            ptr,
            layout
        );

        return tensor;
    }, 
    py::arg("data"),
    py::arg("n") = 1,
    py::arg("h") = 1,
    py::arg("w") = 1,
    py::arg("c") = 1,
    py::arg("layout") = TensorLayout::Layout::LINEAR,
    "Factory function to create VulkanTensor with data and dimensions.");

    // -----------------------------
    // Element-wise Addition Operation
    // -----------------------------
    m.def("vulkan_add", [](py::array_t<float> inputA, py::array_t<float> inputB, py::array_t<float> output) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input and output arrays are contiguous
            if (!(inputA.flags() & py::array::c_style) ||
                !(inputB.flags() & py::array::c_style) ||
                !(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Input and output arrays must be contiguous.");
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
                1, 1, 1, 1, // w, h, c, n
                buf_a.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorB(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_b.size * sizeof(float)),
                1, 1, 1, 1, // w, h, c, n
                buf_b.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorC(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_c.size * sizeof(float)),
                1, 1, 1, 1, // w, h, c, n
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
    }, 
    py::arg("inputA"), py::arg("inputB"), py::arg("output"),
    "Execute element-wise addition on Vulkan backend");

    // -----------------------------
    // Matrix Multiplication Operation
    // -----------------------------
    m.def("vulkan_matmul", [](py::array_t<float> inputA, py::array_t<float> inputB, py::array_t<float> output,
                              uint32_t M, uint32_t K, uint32_t N) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input and output arrays are contiguous
            if (!(inputA.flags() & py::array::c_style) ||
                !(inputB.flags() & py::array::c_style) ||
                !(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Input and output arrays must be contiguous.");
            }

            auto buf_a = inputA.request();
            auto buf_b = inputB.request();
            auto buf_c = output.request();

            // Calculate expected sizes
            size_t expected_size_a = static_cast<size_t>(M) * K;
            size_t expected_size_b = static_cast<size_t>(K) * N;
            size_t expected_size_c = static_cast<size_t>(M) * N;

            if (buf_a.size != expected_size_a ||
                buf_b.size != expected_size_b ||
                buf_c.size != expected_size_c) {
                throw std::runtime_error("Input and output array sizes do not match the specified dimensions.");
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorA(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_a.size * sizeof(float)),
                K, M, 1, 1, // w, h, c, n
                buf_a.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorB(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_b.size * sizeof(float)),
                N, K, 1, 1, // w, h, c, n
                buf_b.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorC(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_c.size * sizeof(float)),
                N, M, 1, 1, // w, h, c, n
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
    }, 
    py::arg("inputA"), py::arg("inputB"), py::arg("output"), 
    py::arg("M"), py::arg("K"), py::arg("N"),
    "Execute matrix multiplication on Vulkan backend");

    // -----------------------------
    // ReLU Operation
    // -----------------------------
    m.def("vulkan_relu", [](py::array_t<float> input, py::array_t<float> output) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input and output arrays are contiguous
            if (!(input.flags() & py::array::c_style) ||
                !(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Input and output arrays must be contiguous.");
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
                1, 1, 1, 1, // w, h, c, n
                buf_input.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                1, 1, 1, 1, // w, h, c, n
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
    }, 
    py::arg("input"), py::arg("output"),
    "Execute ReLU activation on Vulkan backend");

    // -----------------------------
    // Sigmoid Operation
    // -----------------------------
    m.def("vulkan_sigmoid", [](py::array_t<float> input, py::array_t<float> output) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input and output arrays are contiguous
            if (!(input.flags() & py::array::c_style) ||
                !(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Input and output arrays must be contiguous.");
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
                1, 1, 1, 1, // w, h, c, n
                buf_input.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                1, 1, 1, 1, // w, h, c, n
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
    }, 
    py::arg("input"), py::arg("output"),
    "Execute Sigmoid activation on Vulkan backend");

    // -----------------------------
    // Softmax Operation
    // -----------------------------
    m.def("vulkan_softmax", [](py::array_t<float> input, py::array_t<float> output) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input and output arrays are contiguous
            if (!(input.flags() & py::array::c_style) ||
                !(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Input and output arrays must be contiguous.");
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
                1, 1, 1, 1, // w, h, c, n
                buf_input.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                1, 1, 1, 1, // w, h, c, n
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
    }, 
    py::arg("input"), py::arg("output"),
    "Execute Softmax operation on Vulkan backend");

    // -----------------------------
    // Conv2D Operation
    // -----------------------------
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

            // Ensure input, kernel, and output arrays are contiguous
            if (!(input.flags() & py::array::c_style) ||
                !(kernel.flags() & py::array::c_style) ||
                !(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Input, kernel, and output arrays must be contiguous.");
            }

            auto buf_input = input.request();
            auto buf_kernel = kernel.request();
            auto buf_output = output.request();

            uint32_t output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
            uint32_t output_height = (input_height + 2 * padding - kernel_size) / stride + 1;

            // Validate tensor sizes
            if (buf_input.size != static_cast<size_t>(input_width) * input_height * input_channels) {
                throw std::runtime_error("Input size mismatch.");
            }
            if (buf_kernel.size != static_cast<size_t>(kernel_size) * kernel_size * input_channels * output_channels) {
                throw std::runtime_error("Kernel size mismatch.");
            }
            if (buf_output.size != static_cast<size_t>(output_width) * output_height * output_channels) {
                throw std::runtime_error("Output size mismatch.");
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorInput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_input.size * sizeof(float)),
                input_width, input_height, input_channels, 1, // w, h, c, n
                buf_input.ptr,
                TensorLayout::Layout::NHWC
            );
            VulkanTensor tensorKernel(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_kernel.size * sizeof(float)),
                kernel_size, kernel_size, input_channels, output_channels, // w, h, c, n
                buf_kernel.ptr,
                TensorLayout::Layout::NHWC
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                output_width, output_height, output_channels, 1, // w, h, c, n
                nullptr,
                TensorLayout::Layout::NHWC
            );

            // Prepare push constants
            Conv2DPushConstants pushConstants = {
                input_width,
                input_height,
                input_channels,
                output_channels,
                kernel_size,
                padding,
                stride
            };

            // Execute Conv2D operation
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
    py::arg("stride"),
    "Execute 2D Convolution on Vulkan backend");

    // -----------------------------
    // MaxPool Operation
    // -----------------------------
    m.def("vulkan_maxpool", [](py::array_t<float> input, py::array_t<float> output,
                              uint32_t width, uint32_t height, uint32_t channels,
                              uint32_t poolSizeX, uint32_t poolSizeY,
                              uint32_t strideX, uint32_t strideY) {
        try {
            // Ensure Vulkan is initialized
            if (!vulkan_globals::getContext() || vulkan_globals::device == VK_NULL_HANDLE) {
                throw std::runtime_error("Vulkan is not initialized. Call init_vulkan() first.");
            }

            // Ensure input and output arrays are contiguous
            if (!(input.flags() & py::array::c_style) ||
                !(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Input and output arrays must be contiguous.");
            }

            auto buf_input = input.request();
            auto buf_output = output.request();

            uint32_t expected_output_width = (width - poolSizeX) / strideX + 1;
            uint32_t expected_output_height = (height - poolSizeY) / strideY + 1;
            uint32_t expected_output_size = expected_output_width * expected_output_height * channels;

            if (buf_input.size != static_cast<size_t>(width) * height * channels) {
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
                width, height, channels, 1, // w, h, c, n
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
                1, // n
                nullptr,
                TensorLayout::Layout::NHWC
            );

            // Execute MaxPool operation
            vulkan_ops::executeMaxPool(tensorInput, tensorOutput, 
                                       width, height, channels,
                                       poolSizeX, poolSizeY, strideX, strideY);

            // Download result
            tensorOutput.download(buf_output.ptr);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("MaxPool operation failed: " + std::string(e.what()));
        }
    }, 
    py::arg("input"),
    py::arg("output"),
    py::arg("width"),
    py::arg("height"),
    py::arg("channels"),
    py::arg("poolSizeX"),
    py::arg("poolSizeY"),
    py::arg("strideX"),
    py::arg("strideY"),
    "Execute MaxPool2D operation on Vulkan backend");

    // -----------------------------
    // BatchNorm Operation
    // -----------------------------
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
            if (!(input.flags() & py::array::c_style) ||
                !(gamma.flags() & py::array::c_style) ||
                !(beta.flags() & py::array::c_style) ||
                !(output.flags() & py::array::c_style)) {
                throw std::runtime_error("Input, gamma, beta, and output arrays must be contiguous.");
            }

            auto buf_input = input.request();
            auto buf_gamma = gamma.request();
            auto buf_beta = beta.request();
            auto buf_output = output.request();

            if (buf_input.size != size ||
                buf_gamma.size != size ||
                buf_beta.size != size ||
                buf_output.size != size) {
                throw std::runtime_error("All tensors must have the specified size.");
            }

            auto context = vulkan_globals::getContext();

            // Create VulkanTensor instances
            VulkanTensor tensorInput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_input.size * sizeof(float)),
                1, 1, size, 1, // w, h, c, n
                buf_input.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorGamma(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_gamma.size * sizeof(float)),
                1, 1, size, 1, // w, h, c, n
                buf_gamma.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorBeta(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_beta.size * sizeof(float)),
                1, 1, size, 1, // w, h, c, n
                buf_beta.ptr,
                TensorLayout::Layout::LINEAR
            );
            VulkanTensor tensorOutput(
                context->getMemoryManager(),
                context->getBufferPool(),
                checkSize(buf_output.size * sizeof(float)),
                1, 1, size, 1, // w, h, c, n
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
    }, 
    py::arg("input"),
    py::arg("gamma"),
    py::arg("beta"),
    py::arg("output"),
    py::arg("size"),
    py::arg("epsilon"),
    "Execute BatchNorm operation on Vulkan backend");

    // -----------------------------
    // Save Model Operation (Not Implemented)
    // -----------------------------
    m.def("save_model", [](const std::string& filename) -> void {
        throw std::runtime_error("Model saving not yet implemented.");
    }, "Save the current model to a file");

    // -----------------------------
    // Load Model Operation (Not Implemented)
    // -----------------------------
    m.def("load_model", [](const std::string& filename) -> void {
        throw std::runtime_error("Model loading not yet implemented.");
    }, "Load a model from a file");

    // -----------------------------
    // Initialize Distributed Training (Not Implemented)
    // -----------------------------
    m.def("initialize_distributed", [](uint32_t num_gpus) -> void {
        throw std::runtime_error("Distributed training not yet implemented.");
    }, "Initialize distributed training");

    // -----------------------------
    // Enable Gradient Checkpointing (Not Implemented)
    // -----------------------------
    m.def("enable_gradient_checkpointing", []() -> void {
        throw std::runtime_error("Gradient checkpointing not yet implemented.");
    }, "Enable gradient checkpointing");
}
