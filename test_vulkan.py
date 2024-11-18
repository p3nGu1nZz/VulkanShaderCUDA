import numpy as np
import vulkan_backend


def generate_random_data(shape, dtype=np.float32):
    """
    Generate random data of a given shape and dtype.
    """
    return np.random.rand(*shape).astype(dtype)


def test_addition():
    print("Testing Vulkan addition...")

    # Initialize Vulkan backend
    vulkan_backend.init_vulkan()

    # Generate random data
    size = 1024
    a_data = generate_random_data((size,))
    b_data = generate_random_data((size,))
    expected_output = a_data + b_data

    # Allocate Vulkan tensors
    a = vulkan_backend.VulkanTensor(size * 4)  # Each float is 4 bytes
    b = vulkan_backend.VulkanTensor(size * 4)
    c = vulkan_backend.VulkanTensor(size * 4)

    # Upload data to Vulkan tensors
    a.upload(a_data)
    b.upload(b_data)

    # Perform addition
    vulkan_backend.vulkan_add(a, b, c)

    # Download data from Vulkan tensor
    c_data = c.download()

    # Verify results
    assert np.allclose(c_data, expected_output), "Addition test failed!"
    print("Addition test passed!")


def test_matrix_multiplication():
    print("\nTesting Vulkan matrix multiplication...")

    # Generate random data
    M, K, N = 16, 16, 16
    a_data = generate_random_data((M, K))
    b_data = generate_random_data((K, N))
    expected_output = np.dot(a_data, b_data)

    # Allocate Vulkan tensors
    a = vulkan_backend.VulkanTensor(M * K * 4)
    b = vulkan_backend.VulkanTensor(K * N * 4)
    c = vulkan_backend.VulkanTensor(M * N * 4)

    # Upload data to Vulkan tensors
    a.upload(a_data.flatten())
    b.upload(b_data.flatten())

    # Perform matrix multiplication
    vulkan_backend.vulkan_matmul(a, b, c, M, N, K)

    # Download data from Vulkan tensor
    c_data = c.download().reshape(M, N)

    # Verify results
    assert np.allclose(c_data, expected_output), "Matrix multiplication test failed!"
    print("Matrix multiplication test passed!")


def test_relu():
    print("\nTesting Vulkan ReLU activation...")

    # Generate random data
    size = 1024
    input_data = generate_random_data((size,)) - 0.5  # Generate both positive and negative values
    expected_output = np.maximum(input_data, 0)

    # Allocate Vulkan tensors
    input_tensor = vulkan_backend.VulkanTensor(size * 4)
    output_tensor = vulkan_backend.VulkanTensor(size * 4)

    # Upload data to Vulkan tensor
    input_tensor.upload(input_data)

    # Perform ReLU activation
    vulkan_backend.vulkan_relu(input_tensor, output_tensor)

    # Download data from Vulkan tensor
    output_data = output_tensor.download()

    # Verify results
    assert np.allclose(output_data, expected_output), "ReLU test failed!"
    print("ReLU test passed!")


def test_softmax():
    print("\nTesting Vulkan softmax...")

    # Generate random data
    size = 1024
    input_data = generate_random_data((size,))
    exp_data = np.exp(input_data)
    expected_output = exp_data / np.sum(exp_data)

    # Allocate Vulkan tensors
    input_tensor = vulkan_backend.VulkanTensor(size * 4)
    output_tensor = vulkan_backend.VulkanTensor(size * 4)

    # Upload data to Vulkan tensor
    input_tensor.upload(input_data)

    # Perform softmax activation
    vulkan_backend.vulkan_softmax(input_tensor, output_tensor)

    # Download data from Vulkan tensor
    output_data = output_tensor.download()

    # Verify results
    assert np.allclose(output_data, expected_output), "Softmax test failed!"
    print("Softmax test passed!")


def test_convolution():
    print("\nTesting Vulkan 2D convolution...")

    # Generate random data
    input_width, input_height = 8, 8
    filter_width, filter_height = 3, 3
    stride = 1

    input_data = generate_random_data((input_height, input_width))
    filter_data = generate_random_data((filter_height, filter_width))
    output_width = (input_width - filter_width) // stride + 1
    output_height = (input_height - filter_height) // stride + 1

    expected_output = np.zeros((output_height, output_width))
    for y in range(output_height):
        for x in range(output_width):
            region = input_data[
                y * stride : y * stride + filter_height,
                x * stride : x * stride + filter_width,
            ]
            expected_output[y, x] = np.sum(region * filter_data)

    # Allocate Vulkan tensors
    input_tensor = vulkan_backend.VulkanTensor(input_height * input_width * 4)
    filter_tensor = vulkan_backend.VulkanTensor(filter_height * filter_width * 4)
    output_tensor = vulkan_backend.VulkanTensor(output_height * output_width * 4)

    # Upload data to Vulkan tensors
    input_tensor.upload(input_data.flatten())
    filter_tensor.upload(filter_data.flatten())

    # Perform 2D convolution
    vulkan_backend.vulkan_conv2d(
        input_tensor, filter_tensor, output_tensor, input_width, input_height, filter_width, filter_height, stride
    )

    # Download data from Vulkan tensor
    output_data = output_tensor.download().reshape(output_height, output_width)

    # Verify results
    assert np.allclose(output_data, expected_output), "Convolution test failed!"
    print("Convolution test passed!")


if __name__ == "__main__":
    print("Starting Vulkan backend tests...")
    test_addition()
    test_matrix_multiplication()
    test_relu()
    test_softmax()
    test_convolution()
    print("\nAll tests completed successfully!")
