import torch
import numpy as np
import vulkan_backend
import math
from typing import Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_contiguous_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a contiguous NumPy array.
    
    Args:
        tensor (torch.Tensor): Input tensor
        
    Returns:
        np.ndarray: Contiguous NumPy array
    """
    return tensor.detach().contiguous().cpu().numpy()

def validate_tensor_shapes(*tensors: torch.Tensor) -> None:
    """
    Validate that all tensors are contiguous and have compatible data types.
    
    Args:
        *tensors: Variable number of tensors to validate
    
    Raises:
        ValueError: If validation fails
    """
    if not tensors:
        return
    
    base_dtype = tensors[0].dtype
    for i, tensor in enumerate(tensors):
        if not tensor.is_contiguous():
            raise ValueError(f"Tensor {i} must be contiguous")
        if tensor.dtype != base_dtype:
            raise ValueError(f"Tensor {i} has dtype {tensor.dtype}, expected {base_dtype}")

class VulkanAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for element-wise addition.
        
        Args:
            ctx: Context object for backward pass
            a (torch.Tensor): First input tensor
            b (torch.Tensor): Second input tensor
            
        Returns:
            torch.Tensor: Result of element-wise addition
        """
        if a.shape != b.shape:
            raise ValueError(f"Tensors must have the same shape. Got {a.shape} and {b.shape}")
        
        validate_tensor_shapes(a, b)
        c = torch.empty_like(a)
        
        try:
            a_np = to_contiguous_numpy(a)
            b_np = to_contiguous_numpy(b)
            c_np = to_contiguous_numpy(c)
            
            vulkan_backend.vulkan_add(a_np, b_np, c_np)
            c.copy_(torch.from_numpy(c_np))
            
        except Exception as e:
            logger.error(f"Vulkan addition failed: {str(e)}")
            raise
        
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass for element-wise addition.
        
        Args:
            ctx: Context object from forward pass
            grad_output (torch.Tensor): Gradient from subsequent layer
            
        Returns:
            tuple: Gradients for each input tensor
        """
        return grad_output, grad_output

class VulkanMatMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for matrix multiplication.
        
        Args:
            ctx: Context object for backward pass
            a (torch.Tensor): First input tensor
            b (torch.Tensor): Second input tensor
            
        Returns:
            torch.Tensor: Result of matrix multiplication
        """
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError("Both tensors must be 2D")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Inner dimensions must match. Got {a.shape} and {b.shape}")
        
        validate_tensor_shapes(a, b)
        c = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
        
        try:
            a_np = to_contiguous_numpy(a)
            b_np = to_contiguous_numpy(b)
            c_np = to_contiguous_numpy(c)
            
            vulkan_backend.vulkan_matmul(a_np, b_np, c_np)
            c.copy_(torch.from_numpy(c_np))
            
        except Exception as e:
            logger.error(f"Vulkan matrix multiplication failed: {str(e)}")
            raise
        
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass for matrix multiplication.
        
        Args:
            ctx: Context object from forward pass
            grad_output (torch.Tensor): Gradient from subsequent layer
            
        Returns:
            tuple: Gradients for each input tensor
        """
        a, b = ctx.saved_tensors
        grad_a = torch.matmul(grad_output, b.t())
        grad_b = torch.matmul(a.t(), grad_output)
        return grad_a, grad_b

class VulkanReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ReLU activation.
        
        Args:
            ctx: Context object for backward pass
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Result of ReLU activation
        """
        validate_tensor_shapes(x)
        output = torch.empty_like(x)
        
        try:
            x_np = to_contiguous_numpy(x)
            output_np = to_contiguous_numpy(output)
            
            vulkan_backend.vulkan_relu(x_np, output_np, x.numel())
            output.copy_(torch.from_numpy(output_np))
            
        except Exception as e:
            logger.error(f"Vulkan ReLU failed: {str(e)}")
            raise
        
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass for ReLU activation.
        
        Args:
            ctx: Context object from forward pass
            grad_output (torch.Tensor): Gradient from subsequent layer
            
        Returns:
            torch.Tensor: Gradient for input tensor
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

class VulkanSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Sigmoid activation.
        
        Args:
            ctx: Context object for backward pass
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Result of Sigmoid activation
        """
        validate_tensor_shapes(x)
        output = torch.empty_like(x)
        
        try:
            x_np = to_contiguous_numpy(x)
            output_np = to_contiguous_numpy(output)
            
            vulkan_backend.vulkan_sigmoid(x_np, output_np, x.numel())
            output.copy_(torch.from_numpy(output_np))
            
        except Exception as e:
            logger.error(f"Vulkan Sigmoid failed: {str(e)}")
            raise
        
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass for Sigmoid activation.
        
        Args:
            ctx: Context object from forward pass
            grad_output (torch.Tensor): Gradient from subsequent layer
            
        Returns:
            torch.Tensor: Gradient for input tensor
        """
        output, = ctx.saved_tensors
        return grad_output * output * (1 - output)

class VulkanSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Softmax activation.
        
        Args:
            ctx: Context object for backward pass
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Result of Softmax activation
        """
        validate_tensor_shapes(x)
        output = torch.empty_like(x)
        
        try:
            x_np = to_contiguous_numpy(x)
            output_np = to_contiguous_numpy(output)
            
            vulkan_backend.vulkan_softmax(x_np, output_np, x.numel())
            output.copy_(torch.from_numpy(output_np))
            
        except Exception as e:
            logger.error(f"Vulkan Softmax failed: {str(e)}")
            raise
        
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass for Softmax activation.
        
        Args:
            ctx: Context object from forward pass
            grad_output (torch.Tensor): Gradient from subsequent layer
            
        Returns:
            torch.Tensor: Gradient for input tensor
        """
        output, = ctx.saved_tensors
        return grad_output * output * (1 - torch.sum(output * grad_output, dim=-1, keepdim=True))

class VulkanMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, kernel_size: Tuple[int, int]) -> torch.Tensor:
        """
        Forward pass for Max Pooling 2D.
        
        Args:
            ctx: Context object for backward pass
            x (torch.Tensor): Input tensor (N, C, H, W)
            kernel_size (Tuple[int, int]): Pooling window size
            
        Returns:
            torch.Tensor: Result of max pooling operation
        """
        if x.dim() != 4:
            raise ValueError("Input must be a 4D tensor (N, C, H, W)")
        
        validate_tensor_shapes(x)
        
        N, C, H, W = x.shape
        kernel_h, kernel_w = kernel_size
        output_h = H // kernel_h
        output_w = W // kernel_w
        
        output = torch.empty((N, C, output_h, output_w), device=x.device, dtype=x.dtype)
        
        try:
            for n in range(N):
                x_np = to_contiguous_numpy(x[n])
                output_np = to_contiguous_numpy(output[n])
                
                vulkan_backend.vulkan_pooling(
                    x_np.reshape(H, W, C),
                    output_np.reshape(output_h, output_w, C),
                    W, H, C,
                    kernel_w, kernel_h
                )
                
                output[n].copy_(torch.from_numpy(output_np))
                
        except Exception as e:
            logger.error(f"Vulkan MaxPool2d failed: {str(e)}")
            raise
        
        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass for Max Pooling 2D.
        
        Args:
            ctx: Context object from forward pass
            grad_output (torch.Tensor): Gradient from subsequent layer
            
        Returns:
            tuple: Gradient for input tensor and None for kernel_size
        """
        x, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        
        grad_input = torch.zeros_like(x)
        N, C, H, W = x.shape
        kernel_h, kernel_w = kernel_size
        
        for n in range(N):
            for c in range(C):
                for i in range(0, H, kernel_h):
                    for j in range(0, W, kernel_w):
                        patch = x[n, c, i:i+kernel_h, j:j+kernel_w]
                        max_idx = torch.argmax(patch.view(-1))
                        h_idx = i + max_idx // kernel_w
                        w_idx = j + max_idx % kernel_w
                        grad_input[n, c, h_idx, w_idx] = grad_output[n, c, i//kernel_h, j//kernel_w]
        
        return grad_input, None

class VulkanConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for 2D Convolution.
        
        Args:
            ctx: Context object for backward pass
            x (torch.Tensor): Input tensor (N, C, H, W)
            weight (torch.Tensor): Convolution kernels
            bias (Optional[torch.Tensor]): Bias tensor
            
        Returns:
            torch.Tensor: Result of convolution operation
        """
        if x.dim() != 4:
            raise ValueError("Input must be a 4D tensor (N, C, H, W)")
        if weight.dim() != 4:
            raise ValueError("Weight must be a 4D tensor (out_channels, in_channels, kH, kW)")
        
        validate_tensor_shapes(x, weight)
        if bias is not None:
            validate_tensor_shapes(bias)
        
        N, C, H, W = x.shape
        out_channels, in_channels, kH, kW = weight.shape
        
        if C != in_channels:
            raise ValueError(f"Input channels ({C}) doesn't match weight channels ({in_channels})")
        
        output_h = H - kH + 1
        output_w = W - kW + 1
        output = torch.empty((N, out_channels, output_h, output_w), device=x.device, dtype=x.dtype)
        
        try:
            for n in range(N):
                x_np = to_contiguous_numpy(x[n])
                weight_np = to_contiguous_numpy(weight)
                output_np = to_contiguous_numpy(output[n])
                
                vulkan_backend.vulkan_conv2d(
                    x_np.reshape(H, W, C),
                    weight_np.reshape(out_channels, kH, kW, C),
                    output_np.reshape(output_h, output_w, out_channels),
                    W, H, C,
                    kW, kH, out_channels
                )
                
                output[n].copy_(torch.from_numpy(output_np))
            
            if bias is not None:
                output += bias.view(1, -1, 1, 1)
                
        except Exception as e:
            logger.error(f"Vulkan Conv2d failed: {str(e)}")
            raise
        
        ctx.save_for_backward(x, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass for 2D Convolution.
        
        Args:
            ctx: Context object from forward pass
            grad_output (torch.Tensor): Gradient from subsequent layer
            
        Returns:tuple: Gradients for input, weight, and bias tensors
        """
        x, weight, bias = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum((0, 2, 3))
        
        grad_input = torch.nn.grad.conv2d_input(x.shape, weight, grad_output, stride=1, padding=0)
        grad_weight = torch.nn.grad.conv2d_weight(x, weight.shape, grad_output, stride=1, padding=0)
        
        return grad_input, grad_weight, grad_bias

# Functional interfaces
def vulkan_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise addition using Vulkan."""
    return VulkanAddFunction.apply(a, b)

def vulkan_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using Vulkan."""
    return VulkanMatMulFunction.apply(a, b)

def vulkan_relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU activation using Vulkan."""
    return VulkanReLUFunction.apply(x)

def vulkan_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation using Vulkan."""
    return VulkanSigmoidFunction.apply(x)

def vulkan_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax activation using Vulkan."""
    return VulkanSoftmaxFunction.apply(x)

def vulkan_max_pool2d(x: torch.Tensor, kernel_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    """Max pooling using Vulkan."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    return VulkanMaxPool2dFunction.apply(x, kernel_size)

def vulkan_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """2D convolution using Vulkan."""
    return VulkanConv2dFunction.apply(x, weight, bias)

# PyTorch Module wrappers
class VulkanConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return vulkan_conv2d(x, self.weight, self.bias)

class VulkanMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return vulkan_max_pool2d(x, self.kernel_size)

def create_test_case(shape: tuple, dtype=torch.float32, requires_grad=True) -> torch.Tensor:
    """
    Create a test tensor with reproducible random values.
    
    Args:
        shape (tuple): Shape of the tensor
        dtype (torch.dtype): Data type of the tensor
        requires_grad (bool): Whether the tensor requires gradients
        
    Returns:
        torch.Tensor: Test tensor
    """
    torch.manual_seed(42)  # For reproducibility
    return torch.randn(*shape, dtype=dtype, requires_grad=requires_grad)

def test_vulkan_operations():
    """
    Test all Vulkan operations against PyTorch native implementations.
    """
    logger.info("\nTesting Vulkan Operations vs PyTorch Native")
    logger.info("-" * 50)
    
    def verify_close(vulkan_result, torch_result, name):
        max_diff = torch.max(torch.abs(vulkan_result - torch_result)).item()
        is_close = torch.allclose(vulkan_result, torch_result, rtol=1e-5, atol=1e-5)
        logger.info(f"{name:.<30} {'✓' if is_close else '✗'}")
        logger.info(f"Max difference: {max_diff:.6f}")
        if not is_close:
            logger.warning("First few values:")
            logger.warning(f"Vulkan: {vulkan_result.flatten()[:5]}")
            logger.warning(f"PyTorch: {torch_result.flatten()[:5]}")
        logger.info("")
        return is_close
    
    all_passed = True
    
    try:
        # Test Addition
        logger.info("Testing Addition")
        a = create_test_case((1024,))
        b = create_test_case((1024,))
        vulkan_result = vulkan_add(a, b)
        torch_result = a + b
        all_passed &= verify_close(vulkan_result, torch_result, "Addition")
        
        # Test MatMul
        logger.info("Testing Matrix Multiplication")
        a = create_test_case((32, 64))
        b = create_test_case((64, 32))
        vulkan_result = vulkan_matmul(a, b)
        torch_result = torch.matmul(a, b)
        all_passed &= verify_close(vulkan_result, torch_result, "MatMul")
        
        # Test ReLU
        logger.info("Testing ReLU")
        x = create_test_case((1024,))
        vulkan_result = vulkan_relu(x)
        torch_result = torch.relu(x)
        all_passed &= verify_close(vulkan_result, torch_result, "ReLU")
        
        # Test Sigmoid
        logger.info("Testing Sigmoid")
        x = create_test_case((1024,))
        vulkan_result = vulkan_sigmoid(x)
        torch_result = torch.sigmoid(x)
        all_passed &= verify_close(vulkan_result, torch_result, "Sigmoid")
        
        # Test Softmax
        logger.info("Testing Softmax")
        x = create_test_case((1024,))
        vulkan_result = vulkan_softmax(x)
        torch_result = torch.softmax(x, dim=0)
        all_passed &= verify_close(vulkan_result, torch_result, "Softmax")
        
        # Test MaxPool2d
        logger.info("Testing MaxPool2d")
        x = create_test_case((1, 3, 32, 32))
        pool_vulkan = VulkanMaxPool2d(kernel_size=2)
        pool_torch = torch.nn.MaxPool2d(kernel_size=2)
        vulkan_result = pool_vulkan(x)
        torch_result = pool_torch(x)
        all_passed &= verify_close(vulkan_result, torch_result, "MaxPool2d")
        
        # Test Conv2d
        logger.info("Testing Conv2d")
        x = create_test_case((1, 3, 32, 32))
        conv_vulkan = VulkanConv2d(3, 16, kernel_size=3)
        conv_torch = torch.nn.Conv2d(3, 16, kernel_size=3)
        
        # Use same weights for fair comparison
        conv_torch.weight.data.copy_(conv_vulkan.weight.data)
        if conv_vulkan.bias is not None:
            conv_torch.bias.data.copy_(conv_vulkan.bias.data)
        
        vulkan_result = conv_vulkan(x)
        torch_result = conv_torch(x)
        all_passed &= verify_close(vulkan_result, torch_result, "Conv2d")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        all_passed = False
    
    return all_passed

def test_gradients():
    """
    Test gradients computation for all Vulkan operations.
    """
    logger.info("\nTesting Gradients")
    logger.info("-" * 50)
    
    def verify_gradients(vulkan_fn, torch_fn, inputs, name):
        # Forward pass
        vulkan_output = vulkan_fn(*inputs)
        torch_output = torch_fn(*inputs)
        
        # Backward pass
        vulkan_output.sum().backward()
        torch_output.sum().backward()
        
        # Compare gradients
        all_close = True
        for i, inp in enumerate(inputs):
            if inp.grad is not None:
                vulkan_grad = inp.grad.clone()
                inp.grad = None  # Reset for torch backward
                
                torch_output.sum().backward()
                torch_grad = inp.grad
                
                max_diff = torch.max(torch.abs(vulkan_grad - torch_grad)).item()
                is_close = torch.allclose(vulkan_grad, torch_grad, rtol=1e-5, atol=1e-5)
                logger.info(f"{name} Gradient {i}:".ljust(30) + ('✓' if is_close else '✗'))
                logger.info(f"Max gradient difference: {max_diff:.6f}")
                if not is_close:
                    logger.warning("First few gradient values:")
                    logger.warning(f"Vulkan: {vulkan_grad.flatten()[:5]}")
                    logger.warning(f"PyTorch: {torch_grad.flatten()[:5]}")
                logger.info("")
                all_close &= is_close
        
        return all_close
    
    all_passed = True
    
    try:
        # Test Addition gradients
        logger.info("Testing Addition Gradients")
        a = create_test_case((1024,))
        b = create_test_case((1024,))
        all_passed &= verify_gradients(
            vulkan_add,
            lambda x, y: x + y,
            [a, b],
            "Addition"
        )
        
        # Test MatMul gradients
        logger.info("Testing MatMul Gradients")
        a = create_test_case((32, 64))
        b = create_test_case((64, 32))
        all_passed &= verify_gradients(
            vulkan_matmul,
            torch.matmul,
            [a, b],
            "MatMul"
        )
        
        # Test ReLU gradients
        logger.info("Testing ReLU Gradients")
        x = create_test_case((1024,))
        all_passed &= verify_gradients(
            vulkan_relu,
            torch.relu,
            [x],
            "ReLU"
        )
        
        # Test Sigmoid gradients
        logger.info("Testing Sigmoid Gradients")
        x = create_test_case((1024,))
        all_passed &= verify_gradients(
            vulkan_sigmoid,
            torch.sigmoid,
            [x],
            "Sigmoid"
        )
        
        # Test Softmax gradients
        logger.info("Testing Softmax Gradients")
        x = create_test_case((1024,))
        all_passed &= verify_gradients(
            vulkan_softmax,
            lambda x: torch.softmax(x, dim=0),
            [x],
            "Softmax"
        )
        
        # Test Conv2d gradients
        logger.info("Testing Conv2d Gradients")
        x = create_test_case((1, 3, 32, 32))
        conv = VulkanConv2d(3, 16, kernel_size=3)
        conv_torch = torch.nn.Conv2d(3, 16, kernel_size=3)
        conv_torch.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            conv_torch.bias.data.copy_(conv.bias.data)
        
        all_passed &= verify_gradients(
            lambda x: conv(x),
            lambda x: conv_torch(x),
            [x],
            "Conv2d"
        )
        
    except Exception as e:
        logger.error(f"Gradient test failed with error: {str(e)}")
        all_passed = False
    
    return all_passed

def test_memory_management():
    """
    Test memory management and potential memory leaks.
    """
    logger.info("\nTesting Memory Management")
    logger.info("-" * 50)
    
    try:
        # Repeatedly create and destroy tensors
        for i in range(1000):
            if i % 100 == 0:
                logger.info(f"Memory test iteration {i}/1000")
            a = create_test_case((1024, 1024))
            b = create_test_case((1024, 1024))
            c = vulkan_add(a, b)
            d = vulkan_matmul(a, b)
            loss = (c.sum() + d.sum()) / 2
            loss.backward()
        logger.info("Memory test passed ✓")
        return True
    except Exception as e:
        logger.error(f"Memory test failed: {str(e)} ✗")
        return False

def run_benchmarks():
    """
    Run performance benchmarks comparing Vulkan vs PyTorch native operations.
    """
    logger.info("\nRunning Performance Benchmarks")
    logger.info("-" * 50)
    
    import time
    
    def benchmark(vulkan_fn, torch_fn, inputs, name, num_iterations=100):
        # Warmup
        for _ in range(5):
            vulkan_fn(*inputs)
            torch_fn(*inputs)
        
        # Benchmark Vulkan
        vulkan_start = time.perf_counter()
        for _ in range(num_iterations):
            vulkan_fn(*inputs)
        vulkan_time = (time.perf_counter() - vulkan_start) / num_iterations
        
        # Benchmark PyTorch
        torch_start = time.perf_counter()
        for _ in range(num_iterations):
            torch_fn(*inputs)
        torch_time = (time.perf_counter() - torch_start) / num_iterations
        
        logger.info(f"{name} Performance:")
        logger.info(f"Vulkan: {vulkan_time*1000:.2f}ms")
        logger.info(f"PyTorch: {torch_time*1000:.2f}ms")
        speedup = torch_time / vulkan_time
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info("")
        
        return vulkan_time, torch_time
    
    try:
        # Benchmark Addition
        a = create_test_case((8192, 8192))
        b = create_test_case((8192, 8192))
        benchmark(
            vulkan_add,
            lambda x, y: x + y,
            [a, b],
            "Addition"
        )
        
        # Benchmark MatMul
        a = create_test_case((512, 512))
        b = create_test_case((512, 512))
        benchmark(
            vulkan_matmul,
            torch.matmul,
            [a, b],
            "Matrix Multiplication"
        )
        
        # Benchmark ReLU
        x = create_test_case((8192, 8192))
        benchmark(
            vulkan_relu,
            torch.relu,
            [x],
            "ReLU"
        )
        
        # Benchmark Conv2d
        x = create_test_case((16, 64, 64, 64))
        conv_vulkan = VulkanConv2d(64, 128, kernel_size=3)
        conv_torch = torch.nn.Conv2d(64, 128, kernel_size=3)
        conv_torch.weight.data.copy_(conv_vulkan.weight.data)
        if conv_vulkan.bias is not None:
            conv_torch.bias.data.copy_(conv_vulkan.bias.data)
            
        benchmark(
            lambda x: conv_vulkan(x),
            lambda x: conv_torch(x),
            [x],
            "Conv2d"
        )
        
        return True
    except Exception as e:
        logger.error(f"Benchmark failed with error: {str(e)}")
        return False

def test_edge_cases():
    """
    Test edge cases and error handling.
    """
    logger.info("\nTesting Edge Cases")
    logger.info("-" * 50)
    
    all_passed = True
    
    try:
        # Test empty tensor
        logger.info("Testing empty tensor handling")
        x = torch.empty(0)
        y = torch.empty(0)
        try:
            result = vulkan_add(x, y)
            logger.error("Expected ValueError for empty tensors")
            all_passed = False
        except ValueError:
            logger.info("Empty tensor handling: ✓")
        
        # Test mismatched shapes
        logger.info("\nTesting mismatched shapes")
        x = create_test_case((10, 20))
        y = create_test_case((20, 30))
        try:
            result = vulkan_add(x, y)
            logger.error("Expected ValueError for mismatched shapes")
            all_passed = False
        except ValueError:
            logger.info("Mismatched shape handling: ✓")
        
        # Test non-contiguous tensor
        logger.info("\nTesting non-contiguous tensor")
        x = create_test_case((100, 100))
        y = x.t()  # Create non-contiguous tensor
        try:
            result = vulkan_add(x, y)
            logger.error("Expected ValueError for non-contiguous tensor")
            all_passed = False
        except ValueError:
            logger.info("Non-contiguous tensor handling: ✓")
        
        # Test mixed data types
        logger.info("\nTesting mixed data types")
        x = create_test_case((10, 10), dtype=torch.float32)
        y = create_test_case((10, 10), dtype=torch.float64)
        try:
            result = vulkan_add(x, y)
            logger.error("Expected ValueError for mixed data types")
            all_passed = False
        except ValueError:
            logger.info("Mixed data types handling: ✓")
        
    except Exception as e:
        logger.error(f"Edge case test failed with error: {str(e)}")
        all_passed = False
    
    return all_passed

if __name__ == "__main__":
    try:
        logger.info("Initializing Vulkan backend...")
        vulkan_backend.init_vulkan()
        
        # Run main tests
        operations_passed = test_vulkan_operations()
        gradients_passed = test_gradients()
        memory_passed = test_memory_management()
        edge_cases_passed = test_edge_cases()
        benchmark_passed = run_benchmarks()
        
        # Print summary
        logger.info("\nTest Summary")
        logger.info("-" * 50)
        logger.info(f"Operations Test:    {'✓' if operations_passed else '✗'}")
        logger.info(f"Gradients Test:     {'✓' if gradients_passed else '✗'}")
        logger.info(f"Memory Test:        {'✓' if memory_passed else '✗'}")
        logger.info(f"Edge Cases Test:    {'✓' if edge_cases_passed else '✗'}")
        logger.info(f"Benchmark Test:     {'✓' if benchmark_passed else '✗'}")
        
        all_passed = (operations_passed and gradients_passed and 
                     memory_passed and edge_cases_passed and benchmark_passed)
        
        logger.info(f"\nOverall Status: {'✓ All tests passed!' if all_passed else '✗ Some tests failed.'}")
        
        if not all_passed:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Tests failed with unexpected error: {str(e)}")
        sys.exit(1)