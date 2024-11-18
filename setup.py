from setuptools import setup, Extension 
from pybind11.setup_helpers import Pybind11Extension 
ext_modules = [ 
    Pybind11Extension( 
        "vulkan_backend", 
        ["vulkan_backend.cpp"], 
        include_dirs=["C:\VulkanSDK\1.3.296.0/Include"], 
        library_dirs=["C:\VulkanSDK\1.3.296.0/Lib"], 
        libraries=["vulkan-1"], 
    ) 
] 
setup( 
    name="vulkan_backend", 
    ext_modules=ext_modules, 
    zip_safe=False, 
) 
