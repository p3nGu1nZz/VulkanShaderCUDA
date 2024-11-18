from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import sys

extra_compile_args = ['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17']

ext_modules = [
    Pybind11Extension(
        "vulkan_backend",
        ["vulkan_backend.cpp"],
        include_dirs=[r"C:\VulkanSDK\1.3.296.0/Include"],
        library_dirs=[r"C:\VulkanSDK\1.3.296.0/Lib"],
        libraries=["vulkan-1"],
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="vulkan_backend",
    ext_modules=ext_modules,
    zip_safe=False,
)
