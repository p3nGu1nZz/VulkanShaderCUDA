
import os
import sys
from pathlib import Path
import logging
import time
from typing import Tuple, Optional, Union
import math

import numpy as np
import torch

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
# Initialize Vulkan
if vulkan_backend.init_vulkan():
    print("Vulkan initialized successfully.")

# Create input tensors
data_a = np.random.rand(2, 3, 4, 5).astype(np.float32)  # Example shape: (n, c, h, w)
data_b = np.random.rand(2, 3, 4, 5).astype(np.float32)
output = np.empty_like(data_a)

print("Input Tensor A:", data_a)
print("Input Tensor B:", data_b)

# Perform addition
vulkan_backend.vulkan_add(data_a, data_b, output)
print("Output Tensor (Result):", output)

# Cleanup Vulkan
vulkan_backend.cleanup_vulkan()
