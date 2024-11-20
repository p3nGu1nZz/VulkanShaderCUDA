cmake_minimum_required(VERSION 3.15)
project(VulkanShaderCUDA LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Output Directories - Make these more explicit
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib/Debug")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib/Release")

# Create necessary directories
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/shaders")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib/Debug")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib/Release")

# MSVC Settings
if(MSVC)
    add_compile_options(/MP /utf-8 /wd4251 /wd4275 /wd4005 /wd4068)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
endif()

# Find Required Packages
find_package(Vulkan REQUIRED COMPONENTS glslc glslangValidator)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Include FetchContent
include(FetchContent)

# pybind11 Setup with specific configuration
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

# ONNX Setup
set(ONNX_GENERATED_DIR "${CMAKE_BINARY_DIR}/onnx_generated")
file(MAKE_DIRECTORY ${ONNX_GENERATED_DIR})

set(ONNX_PROTOBUF_CC "${ONNX_GENERATED_DIR}/onnx.pb.cc")
set(ONNX_PROTOBUF_H "${ONNX_GENERATED_DIR}/onnx.pb.h")

if(NOT EXISTS ${ONNX_PROTOBUF_CC} OR NOT EXISTS ${ONNX_PROTOBUF_H})
    message(STATUS "Generating ONNX Protobuf files...")
    execute_process(
        COMMAND ${protobuf_BINARY_DIR}/${CMAKE_CFG_INTDIR}/protoc
                --proto_path=${CMAKE_SOURCE_DIR}/onnx/onnx
                --cpp_out=${ONNX_GENERATED_DIR}
                ${CMAKE_SOURCE_DIR}/onnx/onnx/onnx.proto
        RESULT_VARIABLE PROTOC_RESULT
    )
    if(NOT PROTOC_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to generate ONNX protobuf files")
    endif()
endif()

add_custom_target(ONNXProtobuf ALL DEPENDS ${ONNX_PROTOBUF_H} ${ONNX_PROTOBUF_CC})

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
        OUTPUT "${SPIRV_OUTPUT_DIR}/${SHADER_NAME}.spv"
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V ${SHADER} -o "${SPIRV_OUTPUT_DIR}/${SHADER_NAME}.spv"
        DEPENDS ${SHADER}
        COMMENT "Compiling GLSL shader: ${SHADER_NAME}"
    )
    list(APPEND SPIRV_FILES "${SPIRV_OUTPUT_DIR}/${SHADER_NAME}.spv")
endforeach()

add_custom_target(CompileShaders ALL DEPENDS ${SPIRV_FILES})

# Main Target with explicit output name and directory
pybind11_add_module(vulkan_backend MODULE
    src/vulkan_backend.cpp
    ${ONNX_PROTOBUF_CC}
)

# Set explicit output directory for the module
set_target_properties(vulkan_backend PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
    # Ensure correct suffix for Windows
    SUFFIX ".pyd"
)

target_include_directories(vulkan_backend PRIVATE
    ${ONNX_GENERATED_DIR}
    ${abseil_SOURCE_DIR}
    ${spdlog_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${protobuf_SOURCE_DIR}/src
    ${protobuf_BINARY_DIR}
)

target_link_libraries(vulkan_backend PRIVATE
    Vulkan::Vulkan
    absl::base
    absl::strings
    absl::log
    absl::log_internal_message
    absl::log_internal_check_op
    absl::log_internal_conditions
    absl::log_internal_format
    absl::log_internal_globals
    absl::log_internal_proto
    absl::log_internal_nullguard
    absl::log_internal_log_sink_set
    spdlog::spdlog
    protobuf::libprotobuf
)

add_dependencies(vulkan_backend ONNXProtobuf CompileShaders)

# Add custom command to copy the module to final location
add_custom_command(TARGET vulkan_backend POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy 
        "$<TARGET_FILE:vulkan_backend>"
        "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/vulkan_backend.pyd"
    COMMENT "Copying Python module to final location"
)

# Configuration Summary
message(STATUS "Configuration Summary:")
message(STATUS "-------------------")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Vulkan SDK: ${Vulkan_SDK}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ONNX Protobuf Directory: ${ONNX_GENERATED_DIR}")
message(STATUS "Shader output directory: ${SPIRV_OUTPUT_DIR}")
message(STATUS "Module output directory: ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}")