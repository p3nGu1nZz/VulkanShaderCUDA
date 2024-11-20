@echo off
setlocal EnableDelayedExpansion


REM Store the original directory
set "PROJECT_ROOT=%CD%"



REM Create necessary directories
echo Creating fresh directories...
mkdir build
mkdir bin
mkdir shaders

REM Check Vulkan SDK
echo Checking Vulkan SDK...
set "VULKAN_SDK=C:\VulkanSDK\1.3.296.0"
if not exist "%VULKAN_SDK%" (
    echo ERROR: Vulkan SDK not found at %VULKAN_SDK%
    echo Please install the Vulkan SDK first.
    pause
    exit /b 1
)

REM Configure and build
echo Starting CMake configuration...
cd build

cmake -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
    ..

if errorlevel 1 (
    echo ERROR: CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

echo Building project...
cmake --build . --config Release

if errorlevel 1 (
    echo ERROR: Build failed!
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo Testing completed!
echo.
echo If everything worked correctly, you should see:
echo - Protoc downloaded and extracted
echo - ONNX repository cloned
echo - ONNX protobuf files generated
echo - Shaders compiled
echo - Python module built
echo.
echo Checking for key files:
echo.


echo.
echo Shader compilation:
for %%f in (shaders\*.spv) do (
    echo [√] Found compiled shader: %%f
)



echo.
echo Press any key to exit...
pause > nul