@echo off

REM Set paths for Vulkan SDK
set VULKAN_SDK=C:\VulkanSDK\1.3.296.0
set PATH=%VULKAN_SDK%\Bin;%VULKAN_SDK%\Lib;%PATH%
set INCLUDE=%VULKAN_SDK%\Include;%INCLUDE%
set LIB=%VULKAN_SDK%\Lib;%LIB%

REM Create directories if they don't exist
if not exist "shaders" mkdir shaders

REM Create a virtual environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment!
    pause
    exit /b 1
)
call venv\Scripts\activate

REM Upgrade pip, setuptools, and wheel explicitly
echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
    echo Failed to upgrade pip, setuptools, and wheel!
    pause
    exit /b 1
)

REM Install PyTorch and dependencies
echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo Installing dependencies...
python -m pip install pybind11 numpy

REM Compile GLSL shaders
echo Compiling GLSL shaders...
set GLSLANG_VALIDATOR=%VULKAN_SDK%\Bin\glslangValidator.exe

REM Compile shaders and place them in the shaders directory
for %%f in (add.comp mul.comp matmul.comp relu.comp sigmoid.comp softmax.comp conv2d.comp pooling.comp) do (
    echo Compiling %%f...
    %GLSLANG_VALIDATOR% -V %%f -o shaders\%%~nf.spv
    if %errorlevel% neq 0 (
        echo Failed to compile %%f
        pause
        exit /b 1
    )
)

echo Shaders compiled successfully!
pause
goto BuildBackend

:ShaderError
echo Shader compilation failed!
pause
exit /b 1

:BuildBackend
REM Build the Vulkan backend
echo Building Vulkan backend...
(
echo from setuptools import setup
echo from pybind11.setup_helpers import Pybind11Extension
echo import sys
echo.
echo extra_compile_args = ['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17']
echo.
echo ext_modules = [
echo     Pybind11Extension(
echo         "vulkan_backend",
echo         ["vulkan_backend.cpp"],
echo         include_dirs=[r"%VULKAN_SDK%/Include"],
echo         library_dirs=[r"%VULKAN_SDK%/Lib"],
echo         libraries=["vulkan-1"],
echo         extra_compile_args=extra_compile_args,
echo     ^)
echo ]
echo.
echo setup(
echo     name="vulkan_backend",
echo     ext_modules=ext_modules,
echo     zip_safe=False,
echo ^)
) > setup.py

python setup.py build_ext --inplace
if %errorlevel% neq 0 (
    echo Vulkan backend build failed!
    pause
    exit /b 1
)
echo Vulkan backend built successfully!

REM Run tests
echo Running tests...
python test_vulkan.py
if %errorlevel% neq 0 (
    echo Tests failed!
    pause
    exit /b 1
)
echo All tests passed successfully!
pause

REM Clean up temporary files
echo Cleaning up...
del setup.py
deactivate

echo Setup completed successfully!
pause
