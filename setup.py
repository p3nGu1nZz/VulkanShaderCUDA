from setuptools import setup 
from pybind11.setup_helpers import Pybind11Extension 
import os 
 
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__)) 
 
setup( 
    name="vulkan_backend", 
    ext_modules=[ 
        Pybind11Extension( 
            "vulkan_backend", 
            [ 
                os.path.join(PROJECT_ROOT, "src", "vulkan_backend.cpp"), 
                os.path.join(PROJECT_ROOT, "onnx", "onnx", "onnx.pb.cc") 
            ], 
            include_dirs=[ 
                os.path.join(PROJECT_ROOT, "abseil"), 
                os.path.join(PROJECT_ROOT, "protoc-28.3", "protobuf-28.3", "src"), 
                os.path.join(PROJECT_ROOT, "protoc-28.3", "protobuf-28.3", "src", "google", "protobuf"), 
                os.path.join(PROJECT_ROOT, "protoc-28.3", "protobuf-28.3", "upb", "base"), 
                os.path.join(PROJECT_ROOT, "onnx", "onnx"), 
                os.path.join(PROJECT_ROOT, "src") 
            ], 
            library_dirs=[ 
                os.path.join(PROJECT_ROOT, "VulkanSDK", "1.3.296.0", "Lib") 
            ], 
            libraries=["vulkan-1"], 
            define_macros=[("PROTOBUF_USE_DLLS", 1)], 
        ) 
    ], 
    zip_safe=False 
) 
