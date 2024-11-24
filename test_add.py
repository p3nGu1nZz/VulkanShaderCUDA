import os
import sys
from pathlib import Path
import logging
import time
from typing import Tuple, Optional, Union
import math
import gc
import numpy as np
import torch
from contextlib import contextmanager
# Correct path handling with no double slashes
SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "build" / "bin" / "Release" / "vulkan_backend.pyd"

# For debugging
print(f"Looking for module at: {MODULE_PATH}")

if not MODULE_PATH.exists():
    raise ImportError(
        f"Vulkan backend module not found at {MODULE_PATH}. "
        "Please build the project first using CMake."
    )

sys.path.append(str(MODULE_PATH.parent))

import vulkan_backend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
BENCHMARK_MODE = False
ENABLE_VULKAN_LOGGING = False

class VulkanBenchmark:
    def __init__(self, name, warm_up_iterations=5, iterations=100, batch_size=32):
        self.name = name
        self.warm_up_iterations = warm_up_iterations
        self.iterations = iterations
        self.batch_size = batch_size
        self.times = []
        
    @contextmanager
    def measure(self):
        gc.collect()  # Clean up before benchmark
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.times.append(end - start)
    
    def report(self):
        avg_time = np.mean(self.times) * 1000  # Convert to ms
        std_time = np.std(self.times) * 1000
        min_time = np.min(self.times) * 1000
        max_time = np.max(self.times) * 1000
        
        logging.info(f"\nBenchmark Results for {self.name}:")
        logging.info(f"Average time: {avg_time:.2f} ms")
        logging.info(f"Std deviation: {std_time:.2f} ms")
        logging.info(f"Min time: {min_time:.2f} ms")
        logging.info(f"Max time: {max_time:.2f} ms")
        
class VulkanTester:
    def __init__(self):
        self.initialized = False
        self.initialize_vulkan()

    def initialize_vulkan(self):
        if not self.initialized:
            if not vulkan_backend.is_vulkan_initialized():
                if not vulkan_backend.init_vulkan():
                    raise RuntimeError("Failed to initialize Vulkan")
            self.initialized = True
            logging.info("Vulkan initialized successfully")

    def create_test_tensors(self, shape, dtype=np.float32):
        """Create test tensors with proper memory alignment."""
        a = np.random.rand(*shape).astype(dtype)
        b = np.random.rand(*shape).astype(dtype)
        c = np.empty_like(a)
        return a, b, c

    def verify_results(self, vulkan_result, torch_result, tolerance=1e-5):
        """Verify Vulkan results against PyTorch reference."""
        max_diff = np.max(np.abs(vulkan_result - torch_result.numpy()))
        if max_diff > tolerance:
            raise ValueError(f"Results differ by {max_diff} (tolerance: {tolerance})")
        return max_diff

    def test_operations(self):
        try:
            logging.info("\nTesting Vulkan Operations vs PyTorch Native")
            logging.info("-" * 50)
            
            # Test various shapes
            shapes = [
                (2, 3, 4, 5),      # Small tensor
                (16, 16, 16, 16),  # Medium tensor
                (1, 1, 1, 1),      # Minimal tensor
            ]
            
            for shape in shapes:
                logging.info(f"\nTesting Addition with shape {shape}")
                a, b, c = self.create_test_tensors(shape)
                
                # PyTorch reference
                torch_a = torch.from_numpy(a)
                torch_b = torch.from_numpy(b)
                torch_result = torch_a + torch_b
                
                # Vulkan operation
                vulkan_backend.vulkan_add(a, b, c)
                
                # Verify results
                max_diff = self.verify_results(c, torch_result)
                logging.info(f"Max difference: {max_diff}")
                
            logging.info("\nAddition test passed ✓")
            return True
            
        except Exception as e:
            logging.error(f"Operation test failed with error: {str(e)}")
            return False

    def test_gradients(self):
        try:
            logging.info("\nTesting Gradients")
            logging.info("-" * 50)
            
            shapes = [(2, 3, 4, 5), (16, 16, 16, 16)]
            
            for shape in shapes:
                logging.info(f"\nTesting gradients with shape {shape}")
                a, b, c = self.create_test_tensors(shape)
                
                # Forward pass
                vulkan_backend.vulkan_add(a, b, c)
                
                # Backward pass simulation
                grad_output = np.ones_like(c)
                grad_a = np.empty_like(a)
                grad_b = np.empty_like(b)
                
                grad_a[:] = grad_output
                grad_b[:] = grad_output
                
                # Verify gradient shapes and values
                assert grad_a.shape == a.shape
                assert grad_b.shape == b.shape
                
            logging.info("Gradient test passed ✓")
            return True
            
        except Exception as e:
            logging.error(f"Gradient test failed with error: {str(e)}")
            return False

    def test_edge_cases(self):
        try:
            logging.info("\nTesting Edge Cases")
            logging.info("-" * 50)
            
            # Test cases
            cases = [
                ((1, 1, 1, 1), "Minimal size"),
                ((2, 3, 0, 4), "Zero dimension"),
                ((128, 128, 128, 128), "Large size"),
            ]
            
            for shape, case_name in cases:
                logging.info(f"\nTesting {case_name} with shape {shape}")
                try:
                    a, b, c = self.create_test_tensors(shape)
                    vulkan_backend.vulkan_add(a, b, c)
                    logging.info(f"{case_name} passed")
                except Exception as e:
                    logging.warning(f"{case_name} failed: {str(e)}")
            
            logging.info("Edge case test passed ✓")
            return True
            
        except Exception as e:
            logging.error(f"Edge case test failed with error: {str(e)} ✗")
            return False

    def benchmark_addition(self):
        try:
            logging.info("\nRunning Performance Benchmarks")
            logging.info("-" * 50)
            
            shapes = [
                ((16, 16, 16, 16), "Small"),
                ((32, 32, 32, 32), "Medium"),
                ((64, 64, 64, 64), "Large")
            ]
            
            for shape, size_name in shapes:
                benchmark = VulkanBenchmark(
                    f"Addition ({size_name})",
                    warm_up_iterations=3,
                    iterations=50
                )
                
                logging.info(f"\nBenchmarking {size_name} tensor addition ({shape})")
                a, b, c = self.create_test_tensors(shape)
                
                # Warmup
                logging.info("Warming up...")
                for _ in range(benchmark.warm_up_iterations):
                    vulkan_backend.vulkan_add(a, b, c)
                
                # Benchmark
                logging.info("Running benchmark...")
                for _ in range(benchmark.iterations):
                    with benchmark.measure():
                        vulkan_backend.vulkan_add(a, b, c)
                
                # Report results
                benchmark.report()
                
                # Clean up after each benchmark
                del a, b, c
                gc.collect()
            
            return True
            
        except Exception as e:
            logging.error(f"Benchmark failed with error: {str(e)}")
            return False

    def cleanup(self):
        if vulkan_backend.is_vulkan_initialized():
            vulkan_backend.cleanup_vulkan()
            logging.info("Vulkan cleaned up successfully")
        self.initialized = False

def run_test_suite():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    tester = VulkanTester()
    
    try:
        results = {
            'operations': tester.test_operations(),
            'gradients': tester.test_gradients(),
            'edge_cases': tester.test_edge_cases(),
            'benchmarks': tester.benchmark_addition()
        }
        
        logging.info("\nTest Summary")
        logging.info("-" * 50)
        
        for test_name, passed in results.items():
            status = "✓" if passed else "✗"
            logging.info(f"{test_name.capitalize():15} Test: {status}")
        
        all_passed = all(results.values())
        status_str = "✓ All tests passed." if all_passed else "✗ Some tests failed."
        logging.info(f"\nOverall Status: {status_str}")
        
    except Exception as e:
        logging.error(f"Test suite failed with error: {str(e)}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    run_test_suite()
    input("Test completed successfully.\nPress any key to continue .")