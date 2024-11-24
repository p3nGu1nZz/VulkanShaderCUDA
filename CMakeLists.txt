cmake_minimum_required(VERSION 3.30)
project(VulkanShaderCUDA LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Add global settings
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)

# Output Directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release")

# Create necessary directories
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/shaders")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/python")

# MSVC Settings
if(MSVC)
    add_compile_options(/MP /utf-8 /wd4251 /wd4275 /wd4005 /wd4068 /bigobj /EHsc)
    add_definitions(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    # Prevent stripping symbols
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO")
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO")
endif()

# Find Required Packages
find_package(Vulkan REQUIRED COMPONENTS glslc glslangValidator)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Include FetchContent
include(FetchContent)

# pybind11 Setup
set(PYBIND11_PYTHON_VERSION ${Python_VERSION})
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Abseil Setup
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_ENABLE_INSTALL ON)
set(BUILD_TESTING OFF)
FetchContent_Declare(
    abseil
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20230802.1
)
FetchContent_MakeAvailable(abseil)

# Protobuf Setup
set(protobuf_BUILD_TESTS OFF)
set(protobuf_BUILD_SHARED_LIBS OFF)
set(protobuf_MSVC_STATIC_RUNTIME ON)
FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v28.3
)
FetchContent_MakeAvailable(protobuf)

# Zlib Setup
set(ZLIB_BUILD_STATIC_LIBS ON)
set(ZLIB_BUILD_SHARED_LIBS OFF)
set(SKIP_INSTALL_ALL ON)
FetchContent_Declare(
    zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.2.13
)
FetchContent_MakeAvailable(zlib)

# Create custom target to ensure zlib is built
add_custom_target(build_zlib
    COMMAND ${CMAKE_COMMAND} --build . --target zlibstatic --config $<CONFIG>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building zlib"
)

# Define ZLIB path and library
set(ZLIB_INCLUDE_DIR ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
set(ZLIB_LIBRARY "$<TARGET_FILE:zlibstatic>")

# ONNX Setup
FetchContent_Declare(
    onnx
    GIT_REPOSITORY https://github.com/onnx/onnx.git
    GIT_TAG v1.14.1
)
FetchContent_MakeAvailable(onnx)

set(ONNX_GENERATED_DIR "${CMAKE_BINARY_DIR}/onnx_generated")
file(MAKE_DIRECTORY ${ONNX_GENERATED_DIR})
set(ONNX_PROTO_PATH "${onnx_SOURCE_DIR}/onnx")
set(ONNX_PROTO "${ONNX_PROTO_PATH}/onnx.proto")

set(ONNX_GENERATED_SOURCES
    "${ONNX_GENERATED_DIR}/onnx/onnx/onnx.pb.cc"
    "${ONNX_GENERATED_DIR}/onnx/onnx/onnx.pb.h"
)

# Generate protobuf files
add_custom_command(
    OUTPUT ${ONNX_GENERATED_SOURCES}
    COMMAND protobuf::protoc
            --proto_path=${onnx_SOURCE_DIR}
            --cpp_out=${ONNX_GENERATED_DIR}/onnx
            --experimental_allow_proto3_optional
            ${ONNX_PROTO}
    DEPENDS 
        ${ONNX_PROTO}
    COMMENT "Generating ONNX protobuf files"
    VERBATIM
)

add_custom_target(ONNXProtobuf ALL 
    DEPENDS ${ONNX_GENERATED_SOURCES}
)

# Fetch spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.11.0
)
FetchContent_MakeAvailable(spdlog)

# Shader Compilation
file(GLOB GLSL_SHADERS "${CMAKE_SOURCE_DIR}/comp/*.comp")
set(SPIRV_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/shaders")

foreach(SHADER ${GLSL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    add_custom_command(
        OUTPUT "${SPIRV_OUTPUT_DIR}/${SHADER_NAME}.comp.spv"
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V ${SHADER} -o "${SPIRV_OUTPUT_DIR}/${SHADER_NAME}.comp.spv"
        DEPENDS ${SHADER}
        COMMENT "Compiling GLSL shader: ${SHADER_NAME}"
    )
    list(APPEND SPIRV_FILES "${SPIRV_OUTPUT_DIR}/${SHADER_NAME}.spv")
endforeach()

add_custom_target(CompileShaders ALL DEPENDS ${SPIRV_FILES})

# Define source and header files
set(SOURCES
    src/VulkanContext.cpp
    src/CommandBufferManager.cpp
    src/DescriptorSetManager.cpp
    src/OnnxModelParser.cpp
    src/PipelineManager.cpp
    src/ShaderManager.cpp
    src/VulkanBufferPool.cpp
    src/VulkanMemoryManager.cpp
    src/VulkanSync.cpp
    src/vulkan_globals.cpp
    src/VulkanOperations.cpp
	src/VulkanTensor.cpp
    ${ONNX_GENERATED_SOURCES}
)

set(HEADERS
    src/VulkanContext.h
    src/CommandBufferManager.h
    src/DescriptorSetManager.h
    src/OnnxModelParser.h
    src/PipelineManager.h
    src/ShaderManager.h
    src/VulkanBufferPool.h
    src/VulkanError.h
    src/VulkanMemoryManager.h
    src/VulkanSync.h
    src/VulkanTensor.h
    src/PushConstants.h
    src/Utils.h
    src/vulkan_globals.h
	src/VulkanDeviceHelper.h
    src/VulkanOperations.h
)

# Create main library target
add_library(vulkan_backend_lib STATIC 
    ${SOURCES}
    ${HEADERS}
)

# Set include directories for the library
target_include_directories(vulkan_backend_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${ONNX_GENERATED_DIR}
    "${ONNX_GENERATED_DIR}/onnx"
    ${onnx_SOURCE_DIR}
    ${onnx_BINARY_DIR}
    ${abseil_SOURCE_DIR}
    ${spdlog_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
    ${protobuf_SOURCE_DIR}/src
    ${protobuf_BINARY_DIR}
    ${ZLIB_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

# Link library dependencies
target_link_libraries(vulkan_backend_lib PUBLIC
    Vulkan::Vulkan
    absl::base
    absl::strings
    absl::log
    spdlog::spdlog
    protobuf::libprotobuf
    zlibstatic
    pybind11::module
    pybind11::headers
    ${Python_LIBRARIES}
)

# Add Python module target
pybind11_add_module(vulkan_backend MODULE 
    src/vulkan_backend_bindings.cpp
)

# Handle linking with WHOLEARCHIVE
if(MSVC)
    set_target_properties(vulkan_backend PROPERTIES 
        LINK_FLAGS "/VERBOSE /WHOLEARCHIVE:vulkan_backend_lib"
        STATIC_LIBRARY_OPTIONS "/WHOLEARCHIVE"
    )
    target_link_directories(vulkan_backend PRIVATE
        "${CMAKE_BINARY_DIR}/lib/$<CONFIG>"
    )
    target_link_libraries(vulkan_backend PRIVATE
        vulkan_backend_lib
        pybind11::module
        pybind11::headers
        ${Python_LIBRARIES}
        Vulkan::Vulkan
        absl::base
        absl::strings
        absl::log
        spdlog::spdlog
        protobuf::libprotobuf
        zlibstatic
    )
    add_custom_command(TARGET vulkan_backend POST_BUILD
        COMMAND dumpbin /HEADERS "$<TARGET_FILE:vulkan_backend>" > "${CMAKE_BINARY_DIR}/pyd_symbols.txt"
        COMMENT "Dumping symbol table for validation at ${CMAKE_BINARY_DIR}/pyd_symbols.txt"
    )
else()
    target_link_libraries(vulkan_backend PRIVATE
        -Wl,--whole-archive vulkan_backend_lib -Wl,--no-whole-archive
        pybind11::module
        pybind11::headers
        ${Python_LIBRARIES}
        Vulkan::Vulkan
        absl::base
        absl::strings
        absl::log
        spdlog::spdlog
        protobuf::libprotobuf
    )
endif()

# Set output properties for Python module
set_target_properties(vulkan_backend PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
    SUFFIX ".pyd"
)

# Configuration Summary
message(STATUS "Configuration Summary:")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Vulkan SDK: ${Vulkan_SDK}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ONNX Protobuf Directory: ${ONNX_GENERATED_DIR}")
message(STATUS "Shader output directory: ${SPIRV_OUTPUT_DIR}")
message(STATUS "Module output directory: ${FINAL_MODULE_DIR}")
message(STATUS "ZLIB_LIBRARY: ${ZLIB_LIBRARY}")
message(STATUS "ZLIB_INCLUDE_DIR: ${ZLIB_INCLUDE_DIR}")
