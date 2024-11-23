@echo off
setlocal EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%"
echo Setting up environment variables...

set "VULKAN_SDK=C:\VulkanSDK\1.3.296.0"
set "PROTOBUF_DIR=%PROJECT_DIR%\protoc-28.3"
set "ONNX_REPO_PATH=%PROJECT_DIR%\onnx"
set "ABSEIL_PATH=%PROJECT_DIR%\abseil"
set "SPDLOG_PATH=%PROJECT_DIR%\libs\spdlog"
set "PYBIND11_DIR=%PROJECT_DIR%\venv\Lib\site-packages\pybind11"
set "GLSLANG_VALIDATOR=%VULKAN_SDK%\Bin\glslangValidator.exe"

rem Updated to use build directory
set "BUILD_DIR=%PROJECT_DIR%build"
set "BIN_DIR=%BUILD_DIR%\bin\Release"
set "VULKAN_BACKEND_PYD=%BIN_DIR%\vulkan_backend.pyd"

set "PATH=%BIN_DIR%;%VULKAN_SDK%\Bin;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem"
set "PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%"

echo Activating virtual environment...
if not exist "%PROJECT_DIR%\venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

call "%PROJECT_DIR%\venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Error activating virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

echo Checking required files...
if not exist "%BIN_DIR%" (
    echo Error: Release bin directory not found at: %BIN_DIR%
    pause
    exit /b 1
)

if not exist "%VULKAN_BACKEND_PYD%" (
    echo Error: Python module not found at: %VULKAN_BACKEND_PYD%
    pause
    exit /b 1
)

echo.
echo Contents of bin directory:
dir "%BIN_DIR%"
echo.

echo Current environment:
echo -------------------
echo Working Directory: %CD%
echo Project Directory: %PROJECT_DIR%
echo Build Directory: %BUILD_DIR%
echo Bin Directory: %BIN_DIR%
echo PYTHONPATH: %PYTHONPATH%
echo PATH: %PATH%
echo.

echo Running test_vulkan.py...
python "%PROJECT_DIR%\test_add.py"
if errorlevel 1 (
    echo Error running test_add.py
    pause
    exit /b 1
)

echo.
echo Test completed successfully.
pause
exit /b 0