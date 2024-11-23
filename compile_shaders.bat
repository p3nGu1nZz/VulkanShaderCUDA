@echo off
setlocal EnableDelayedExpansion

REM Store the original directory
set "PROJECT_ROOT=%CD%"

REM Create necessary directories
echo Creating shaders directory...
if not exist "shaders" mkdir shaders

REM Check Vulkan SDK
echo Checking Vulkan SDK...
set "VULKAN_SDK=C:\VulkanSDK\1.3.296.0"
if not exist "%VULKAN_SDK%" (
    echo ERROR: Vulkan SDK not found at %VULKAN_SDK%
    echo Please install the Vulkan SDK first.
    pause
    exit /b 1
)

REM Run CMake script to compile shaders
echo Starting shader compilation...
cmake -P CompileShaders.cmake
if errorlevel 1 (
    echo ERROR: Shader compilation failed!
    pause
    exit /b 1
)

echo.
echo Shader compilation completed!
echo.
echo Checking compiled shaders:
echo.
for %%f in (shaders\*.spv) do (
    echo [√] Found compiled shader: %%f
)

echo.
echo Press any key to exit...
pause > nul