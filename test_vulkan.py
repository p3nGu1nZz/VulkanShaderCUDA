import torch
import vulkan_backend
import numpy as np
from typing import Tuple


def to_contiguous_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a contiguous NumPy array."""
    return tensor.detach().contiguous().cpu().numpy()


class VulkanAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError(f"Tensors must have the same shape. Got {a.shape} and {b.shape}")
        
        c = torch.empty_like(a)
        a_np = to_contiguous_numpy(a)
        b_np = to_contiguous_numpy(b)
        c_np = to_contiguous_numpy(c)
        
        vulkan_backend.vulkan_add(a_np, b_np, c_np)
        c.copy_(torch.from_numpy(c_np))
        
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        return grad_output, grad_output


class VulkanReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(input)
        input_np = to_contiguous_numpy(input)
        output_np = to_contiguous_numpy(output)
        
        vulkan_backend.vulkan_relu(input_np, output_np, input.numel())
        output.copy_(torch.from_numpy(output_np))
        
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class VulkanSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(input)
        input_np = to_contiguous_numpy(input)
        output_np = to_contiguous_numpy(output)
        
        vulkan_backend.vulkan_sigmoid(input_np, output_np, input.numel())
        output.copy_(torch.from_numpy(output_np))
        
        ctx.save_for_backward(output)  # Save sigmoid output for backward pass
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, = ctx.saved_tensors
        return grad_output * output * (1 - output)


class VulkanSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(input)
        input_np = to_contiguous_numpy(input)
        output_np = to_contiguous_numpy(output)
        
        vulkan_backend.vulkan_softmax(input_np, output_np, input.numel())
        output.copy_(torch.from_numpy(output_np))
        
        ctx.save_for_backward(output)  # Save softmax output for backward pass
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, = ctx.saved_tensors
        return grad_output * output * (1 - output)


class VulkanPoolingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, pool_size: Tuple[int, int]) -> torch.Tensor:
        if len(input.shape) != 3:
            raise ValueError(f"Input tensor must be 3D (width, height, depth). Got shape {input.shape}")
        
        width, height, depth = input.shape
        pool_size_x, pool_size_y = pool_size
        
        if width % pool_size_x != 0 or height % pool_size_y != 0:
            raise ValueError(f"Input dimensions must be divisible by pool size. Got {width}x{height} and pool size {pool_size}")
        
        output_width = width // pool_size_x
        output_height = height // pool_size_y
        output = torch.empty((output_width, output_height, depth), device=input.device)
        
        input_np = to_contiguous_numpy(input)
        output_np = to_contiguous_numpy(output)
        
        vulkan_backend.vulkan_pooling(
            input_np,
            output_np,
            width,
            height,
            depth,
            pool_size_x,
            pool_size_y,
        )
        output.copy_(torch.from_numpy(output_np))
        
        ctx.save_for_backward(input)
        ctx.pool_size = pool_size
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, = ctx.saved_tensors
        pool_size = ctx.pool_size
        raise NotImplementedError("Backward pooling is not implemented in Vulkan backend.")


class VulkanConv2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        if len(input.shape) != 3 or len(kernel.shape) != 3:
            raise ValueError("Input and kernel must be 3D tensors (width, height, depth)")
        
        input_width, input_height, input_depth = input.shape
        kernel_width, kernel_height, kernel_depth = kernel.shape
        
        if input_depth != kernel_depth:
            raise ValueError(f"Input depth ({input_depth}) must match kernel depth ({kernel_depth})")
        
        output_depth = kernel.shape[0]  # Number of filters
        output_width = input_width - kernel_width + 1
        output_height = input_height - kernel_height + 1
        
        if output_width <= 0 or output_height <= 0:
            raise ValueError("Kernel size is too large for input")
        
        output = torch.empty((output_width, output_height, output_depth), device=input.device)
        input_np = to_contiguous_numpy(input)
        kernel_np = to_contiguous_numpy(kernel)
        output_np = to_contiguous_numpy(output)
        
        vulkan_backend.vulkan_conv2d(
            input_np,
            kernel_np,
            output_np,
            input_width,
            input_height,
            input_depth,
            kernel_width,
            kernel_height,
            output_depth,
        )
        output.copy_(torch.from_numpy(output_np))
        
        ctx.save_for_backward(input, kernel)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, kernel = ctx.saved_tensors
        raise NotImplementedError("Backward convolution is not implemented in Vulkan backend.")


# PyTorch-friendly wrappers for Vulkan operations
def vulkan_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise addition using Vulkan backend."""
    return VulkanAddFunction.apply(a, b)


def vulkan_relu(input: torch.Tensor) -> torch.Tensor:
    """ReLU activation using Vulkan backend."""
    return VulkanReLUFunction.apply(input)


def vulkan_sigmoid(input: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation using Vulkan backend."""
    return VulkanSigmoidFunction.apply(input)


def vulkan_softmax(input: torch.Tensor) -> torch.Tensor:
    """Softmax activation using Vulkan backend."""
    return VulkanSoftmaxFunction.apply(input)


def vulkan_pooling(input: torch.Tensor, pool_size: Tuple[int, int]) -> torch.Tensor:
    """Max pooling using Vulkan backend."""
    return VulkanPoolingFunction.apply(input, pool_size)


def vulkan_conv2d(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """2D convolution using Vulkan backend."""
    return VulkanConv2DFunction.apply(input, kernel)


# Initialize Vulkan backend
vulkan_backend.init_vulkan()


# Test functions
def test_vulkan_operations():
    print("Testing Vulkan-backed PyTorch operations...")
    
    # Test addition
    print("\nTesting addition:")
    a = torch.rand(1024, requires_grad=True)
    b = torch.rand(1024, requires_grad=True)
    c = vulkan_add(a, b)
    loss = c.sum()
    loss.backward()
    print(f"Addition shape: {c.shape}")
    print(f"Gradient shapes: {a.grad.shape}, {b.grad.shape}")
    
    # Test ReLU
    print("\nTesting ReLU:")
    input_relu = torch.randn(1024, requires_grad=True)
    output_relu = vulkan_relu(input_relu)
    loss = output_relu.sum()
    loss.backward()
    print(f"ReLU shape: {output_relu.shape}")
    print(f"Gradient shape: {input_relu.grad.shape}")
    
    # Test Sigmoid
    print("\nTesting Sigmoid:")
    input_sigmoid = torch.randn(1024, requires_grad=True)
    output_sigmoid = vulkan_sigmoid(input_sigmoid)
    loss = output_sigmoid.sum()
    loss.backward()
    print(f"Sigmoid shape: {output_sigmoid.shape}")
    print(f"Gradient shape: {input_sigmoid.grad.shape}")
    
    # Test Softmax
    print("\nTesting Softmax:")
    input_softmax = torch.randn(1024, requires_grad=True)
    output_softmax = vulkan_softmax(input_softmax)
    loss = output_softmax.sum()
    loss.backward()
    print(f"Softmax shape: {output_softmax.shape}")
    print(f"Gradient shape: {input_softmax.grad.shape}")
    
    # Test Pooling
    print("\nTesting Pooling:")
    input_pooling = torch.rand(8, 8, 3)
    output_pooling = vulkan_pooling(input_pooling, (2, 2))
    print(f"Pooling input shape: {input_pooling.shape}")
    print(f"Pooling output shape: {output_pooling.shape}")
    
    # Test Convolution
    print("\nTesting Convolution:")
    input_conv = torch.rand(32, 32, 3)
    kernel_conv = torch.rand(3, 3, 3)
    output_conv = vulkan_conv2d(input_conv, kernel_conv)
    print(f"Conv2D input shape: {input_conv.shape}")
    print(f"Conv2D kernel shape: {kernel_conv.shape}")
    print(f"Conv2D output shape: {output_conv.shape}")


if __name__ == "__main__":
    test_vulkan_operations()