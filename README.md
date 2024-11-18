# VulkanShaderCUDA

**VulkanShaderCUDA** is a project that reimplements PyTorch-like tensor operations using Vulkan for high-performance compute on GPUs. The framework uses Vulkan's compute pipelines to perform core operations such as addition, matrix multiplication, ReLU activation, softmax, 2D convolution, and more. This serves as an alternative to CUDA-based solutions while leveraging Vulkan's cross-platform capabilities.

## **Features**
- 🚀 **Core Tensor Operations**: Includes elementwise addition, multiplication, matrix multiplication, ReLU, sigmoid, softmax, 2D convolution, and pooling operations.
- 🎯 **Vulkan-Based Compute Pipelines**: Fully utilizes Vulkan for GPU-accelerated computations.
- 🧪 **Testing Framework**: Includes Python scripts to test Vulkan operations against NumPy outputs.
- 🖥️ **Cross-Platform**: Designed to run on any Vulkan-capable GPU, making it more accessible than CUDA.
- 🔧 **Custom GLSL Shaders**: Implements core operations as GLSL shaders compiled to SPIR-V for Vulkan execution.

---

## **Getting Started**

### **Prerequisites**
1. **Vulkan SDK**:
   - Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).
   - Add the Vulkan SDK to your environment variables:
     - **Windows**:
       ```bash
       set VULKAN_SDK=C:\Path\To\Vulkan\SDK
       set PATH=%VULKAN_SDK%\Bin;%PATH%
       ```
     - Verify installation with:
       ```bash
       vulkaninfo
       ```

2. **Python Environment**:
   - Install Python 3.10 or later.
   - Install required Python libraries:
     ```bash
     pip install numpy pybind11
     ```

3. **GLSL Shader Compiler**:
   - Ensure `glslangValidator` is available in your `PATH` (comes with Vulkan SDK).

---

### **Project Structure**

```plaintext
VulkanShaderCUDA/
│
├── compile_shaders.sh       # Script to compile GLSL shaders to SPIR-V binaries
├── vulkan_backend.cpp       # Vulkan backend implementation (C++ with Pybind11 bindings)
├── test_vulkan.py           # Python script for testing operations
│
├── add.glsl                 # GLSL shader for elementwise addition
├── mul.glsl                 # GLSL shader for elementwise multiplication
├── matmul.glsl              # GLSL shader for matrix multiplication
├── relu.glsl                # GLSL shader for ReLU activation
├── sigmoid.glsl             # GLSL shader for Sigmoid activation
├── softmax.glsl             # GLSL shader for Softmax activation
├── conv2d.glsl              # GLSL shader for 2D convolution
├── pooling.glsl             # GLSL shader for pooling operations (max/avg)
```

---

### **Setup**

#### 1. **Compile Shaders**
Navigate to the project directory and run the `compile_shaders.sh` script to compile all GLSL shaders into SPIR-V binaries:

```bash
./compile_shaders.sh
```

This will generate SPIR-V binaries (`.spv`) for each GLSL shader.

#### 2. **Build the Vulkan Backend**
Create a `setup.py` file for compiling the C++ Vulkan backend using Pybind11:

```python
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "vulkan_backend",
        ["vulkan_backend.cpp"],
        include_dirs=["C:/Path/To/Vulkan/SDK/Include"],
        library_dirs=["C:/Path/To/Vulkan/SDK/Lib"],
        libraries=["vulkan-1"],
    ),
]

setup(
    name="vulkan_backend",
    ext_modules=ext_modules,
    zip_safe=False,
)
```

Run the setup script to build the Vulkan backend:

```bash
python setup.py build_ext --inplace
```

This generates `vulkan_backend.pyd` in the current directory.

#### 3. **Run the Tests**
Run the Python test suite to verify all Vulkan operations:

```bash
python test_vulkan.py
```

---

### **Available Operations**

| Operation          | GLSL Shader     | Description                          |
|---------------------|-----------------|--------------------------------------|
| **Addition**        | `add.glsl`      | Elementwise addition of two tensors. |
| **Multiplication**  | `mul.glsl`      | Elementwise multiplication of two tensors. |
| **Matrix Multiplication** | `matmul.glsl` | Matrix multiplication of two tensors. |
| **ReLU Activation** | `relu.glsl`     | Applies ReLU activation to the input tensor. |
| **Sigmoid Activation** | `sigmoid.glsl` | Applies Sigmoid activation to the input tensor. |
| **Softmax**         | `softmax.glsl`  | Computes softmax over the input tensor. |
| **2D Convolution**  | `conv2d.glsl`   | Performs 2D convolution on input data. |
| **Pooling**         | `pooling.glsl`  | Performs max or average pooling.      |

---

### **Testing Framework**

The `test_vulkan.py` script validates each Vulkan operation by comparing its output to NumPy computations. Each test performs the following steps:
1. Generate random input tensors.
2. Compute expected output using NumPy.
3. Perform the operation using the Vulkan backend.
4. Compare the Vulkan results with NumPy results.

Example Test Output:
```
Starting Vulkan backend tests...
Testing Vulkan addition...
Addition test passed!

Testing Vulkan matrix multiplication...
Matrix multiplication test passed!

Testing Vulkan ReLU activation...
ReLU test passed!

Testing Vulkan softmax...
Softmax test passed!

Testing Vulkan 2D convolution...
Convolution test passed!

All tests completed successfully!
```

---

### **Why VulkanShaderCUDA?**

- **Cross-Platform**: Vulkan is supported on Windows, Linux, and macOS, unlike CUDA, which is limited to NVIDIA GPUs.
- **Efficiency**: Vulkan's low-level control enables high-performance computations.
- **Expandability**: Easily add custom GLSL shaders for additional operations.
- **PyTorch-Like Operations**: Implements many commonly used operations for deep learning and tensor manipulation.

---

### **Future Enhancements**

- Implement additional PyTorch operations (e.g., BatchNorm, Dropout).
- Add support for batched computations.
- Optimize shader pipelines for larger-scale workloads.
- Explore integration with deep learning frameworks like TensorFlow or PyTorch.

---

### **Contributing**

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

### **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to let me know if you'd like to adjust any specific section!