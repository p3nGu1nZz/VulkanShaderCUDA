@echo off

REM Set paths for Vulkan SDK
set VULKAN_SDK=C:\VulkanSDK\1.3.296.0
set PATH=%VULKAN_SDK%\Bin;%VULKAN_SDK%\Lib;%PATH%
set INCLUDE=%VULKAN_SDK%\Include;%INCLUDE%
set LIB=%VULKAN_SDK%\Lib;%LIB%

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
echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
REM Install dependencies
echo Installing dependencies...
python -m pip install pybind11 numpy
if %errorlevel% neq 0 (
    echo Failed to install dependencies!
    pause
    exit /b 1
)

REM Compile GLSL shaders
echo Compiling GLSL shaders...
set GLSLANG_VALIDATOR=%VULKAN_SDK%\Bin\glslangValidator.exe

%GLSLANG_VALIDATOR% -V add.comp -o add.spv
if %errorlevel% neq 0 goto ShaderError
%GLSLANG_VALIDATOR% -V mul.comp -o mul.spv
if %errorlevel% neq 0 goto ShaderError
%GLSLANG_VALIDATOR% -V matmul.comp -o matmul.spv
if %errorlevel% neq 0 goto ShaderError
%GLSLANG_VALIDATOR% -V relu.comp -o relu.spv
if %errorlevel% neq 0 goto ShaderError
%GLSLANG_VALIDATOR% -V sigmoid.comp -o sigmoid.spv
if %errorlevel% neq 0 goto ShaderError
%GLSLANG_VALIDATOR% -V softmax.comp -o softmax.spv
if %errorlevel% neq 0 goto ShaderError
%GLSLANG_VALIDATOR% -V conv2d.comp -o conv2d.spv
if %errorlevel% neq 0 goto ShaderError
%GLSLANG_VALIDATOR% -V pooling.comp -o pooling.spv
if %errorlevel% neq 0 goto ShaderError

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
echo from setuptools import setup, Extension > setup.py
echo from pybind11.setup_helpers import Pybind11Extension >> setup.py
echo ext_modules = [ >> setup.py
echo     Pybind11Extension( >> setup.py
echo         "vulkan_backend", >> setup.py
echo         ["vulkan_backend.cpp"], >> setup.py
echo         include_dirs=["%VULKAN_SDK%/Include"], >> setup.py
echo         library_dirs=["%VULKAN_SDK%/Lib"], >> setup.py
echo         libraries=["vulkan-1"], >> setup.py
echo     ) >> setup.py
echo ] >> setup.py
echo setup( >> setup.py
echo     name="vulkan_backend", >> setup.py
echo     ext_modules=ext_modules, >> setup.py
echo     zip_safe=False, >> setup.py
echo ) >> setup.py

python setup.py build_ext --inplace
if %errorlevel% neq 0 (
    echo Vulkan backend build failed!
    pause
    exit /b 1
)
echo Vulkan backend built successfully!
pause

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
