@echo off
setlocal EnableDelayedExpansion

REM Store the original directory
set "ORIGINAL_DIR=%CD%"

REM Set paths for Vulkan SDK
echo Checking Vulkan SDK installation...
set VULKAN_SDK=C:\VulkanSDK\1.3.296.0
if not exist "%VULKAN_SDK%" (
    echo Error: Vulkan SDK not found at %VULKAN_SDK%
    echo Please install the Vulkan SDK or update the path.
    pause
    exit /b 1
)

echo Vulkan SDK found at %VULKAN_SDK%
set PATH=%VULKAN_SDK%\Bin;%VULKAN_SDK%\Lib;%PATH%
set INCLUDE=%VULKAN_SDK%\Include;%INCLUDE%
set LIB=%VULKAN_SDK%\Lib;%LIB%

echo.
echo Press any key to create virtual environment...
pause > nul

REM Create directories if they don't exist
if not exist "shaders" mkdir shaders

REM Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment!
    pause
    exit /b 1
)

call venv\Scripts\activate
echo Virtual environment created and activated.
echo.
echo Press any key to install dependencies...
pause > nul

REM Upgrade pip and install dependencies
echo Installing pip and basic dependencies...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo Failed to upgrade pip!
    pause
    exit /b 1
)

echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo Failed to install PyTorch!
    pause
    exit /b 1
)

echo Installing other dependencies...
python -m pip install pybind11 numpy onnx protobuf==3.20.2
if errorlevel 1 (
    echo Failed to install dependencies!
    pause
    exit /b 1
)

REM Check for Protocol Buffers Compiler (protoc)
echo.
echo Checking for Protocol Buffers Compiler...
set PROTOC_VERSION=28.3
set PROJECT_BIN=%CD%\bin
set PROTOC_EXE=%PROJECT_BIN%\protoc.exe
set PROTOC_DIR=%CD%\protoc-%PROTOC_VERSION%

echo Looking for protoc at: %PROTOC_EXE%
if not exist "%PROTOC_EXE%" (
    echo Protocol Buffers Compiler not found. Downloading protoc...
    
    REM Create necessary directories
    if not exist "%PROJECT_BIN%" mkdir "%PROJECT_BIN%"
    if not exist "%PROTOC_DIR%" mkdir "%PROTOC_DIR%"
    
    echo Downloading protoc binary...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/protocolbuffers/protobuf/releases/download/v%PROTOC_VERSION%/protoc-%PROTOC_VERSION%-win64.zip' -OutFile 'protoc.zip'}"
    if errorlevel 1 (
        echo Failed to download protoc!
        pause
        exit /b 1
    )
    
    echo Downloading protobuf source...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/protocolbuffers/protobuf/archive/refs/tags/v%PROTOC_VERSION%.zip' -OutFile 'protobuf-src.zip'}"
    if errorlevel 1 (
        echo Failed to download protobuf source!
        pause
        exit /b 1
    )
    
    echo Extracting protoc...
    powershell -Command "& {Expand-Archive -Path 'protoc.zip' -DestinationPath '%PROTOC_DIR%' -Force}"
    if errorlevel 1 (
        echo Failed to extract protoc!
        pause
        exit /b 1
    )
    
    echo Extracting protobuf source...
    powershell -Command "& {Expand-Archive -Path 'protobuf-src.zip' -DestinationPath '%PROTOC_DIR%\src' -Force}"
    if errorlevel 1 (
        echo Failed to extract protobuf source!
        pause
        exit /b 1
    )
    
    REM Move protoc.exe to bin directory
    echo Moving protoc.exe to bin directory...
    copy "%PROTOC_DIR%\bin\protoc.exe" "%PROTOC_EXE%" > nul
    
    REM Set up include directory structure
    echo Setting up include directory structure...
    if not exist "%PROTOC_DIR%\include\google" mkdir "%PROTOC_DIR%\include\google"
    xcopy /E /I /Y "%PROTOC_DIR%\src\protobuf-%PROTOC_VERSION%\src\google" "%PROTOC_DIR%\include\google"
    
    REM Verify critical files
    if not exist "%PROTOC_DIR%\include\google\protobuf\runtime_version.h" (
        echo Error: Failed to find runtime_version.h after setup!
        dir "%PROTOC_DIR%\include\google\protobuf"
        pause
        exit /b 1
    )
    
    REM Cleanup
    echo Cleaning up temporary files...
    if exist "protoc.zip" del "protoc.zip"
    if exist "protobuf-src.zip" del "protobuf-src.zip"
    
    if not exist "%PROTOC_EXE%" (
        echo Failed to setup protoc!
        pause
        exit /b 1
    )
    echo Protocol Buffers Compiler installed successfully.
) else (
    echo Found existing Protocol Buffers Compiler at %PROTOC_EXE%
    
    REM Verify include directory even if protoc exists
    if not exist "%PROTOC_DIR%\include\google\protobuf\runtime_version.h" (
        echo Protobuf include files missing. Downloading source...
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/protocolbuffers/protobuf/archive/refs/tags/v%PROTOC_VERSION%.zip' -OutFile 'protobuf-src.zip'}"
        powershell -Command "& {Expand-Archive -Path 'protobuf-src.zip' -DestinationPath '%PROTOC_DIR%\src' -Force}"
        
        if not exist "%PROTOC_DIR%\include\google" mkdir "%PROTOC_DIR%\include\google"
        xcopy /E /I /Y "%PROTOC_DIR%\src\protobuf-%PROTOC_VERSION%\src\google" "%PROTOC_DIR%\include\google"
        
        if exist "protobuf-src.zip" del "protobuf-src.zip"
    )
)

REM Add project bin and protoc includes to PATH and INCLUDE
set PATH=%PROJECT_BIN%;%PATH%
set INCLUDE=%PROTOC_DIR%\include;%INCLUDE%

echo.
echo Protobuf setup complete. Include directory contents:
dir "%PROTOC_DIR%\include\google\protobuf"
echo.
echo Press any key to continue with ONNX setup...
pause > nul

REM Setup ONNX
echo Setting up ONNX...
set ONNX_REPO_PATH=%CD%\onnx
if not exist "%ONNX_REPO_PATH%" (
    echo Cloning ONNX repository...
    git clone --recursive https://github.com/onnx/onnx.git "%ONNX_REPO_PATH%"
    if errorlevel 1 (
        echo Failed to clone ONNX repository!
        pause
        exit /b 1
    )
)

echo.
echo Generating ONNX Protobuf files...
echo Using protoc from: %PROTOC_EXE%
echo Current directory: %CD%

pushd "%ONNX_REPO_PATH%\onnx"
echo Working directory: %CD%
echo Checking for onnx.proto...
if exist "onnx.proto" (
    echo Found onnx.proto
) else (
    echo ERROR: onnx.proto not found!
    dir
    pause
    exit /b 1
)

"%PROTOC_EXE%" --proto_path=. --cpp_out=. onnx.proto
if errorlevel 1 (
    echo Failed to generate ONNX Protobuf files!
    echo Error code: %errorlevel%
    pause
    exit /b 1
)

echo Checking generated files...
if exist "onnx.pb.h" (
    echo Generated onnx.pb.h successfully
) else (
    echo Failed to generate onnx.pb.h
)
if exist "onnx.pb.cc" (
    echo Generated onnx.pb.cc successfully
) else (
    echo Failed to generate onnx.pb.cc
)

popd

echo.
echo Press any key to compile GLSL shaders...
pause > nul

REM Compile GLSL shaders
echo Compiling GLSL shaders...
set GLSLANG_VALIDATOR=%VULKAN_SDK%\Bin\glslangValidator.exe

REM Create shaders output directory if it doesn't exist
if not exist "shaders" mkdir shaders

REM Check for shader files in comp directory
echo Looking for shader files in: %CD%\comp
if not exist "%CD%\comp\*.comp" (
    echo Error: No shader files found in comp directory at %CD%\comp
    dir "%CD%\comp"
    pause
    exit /b 1
)

REM Compile shaders from comp directory and place them in the shaders directory
for %%f in (comp\*.comp) do (
    echo Compiling %%f...
    %GLSLANG_VALIDATOR% -V "%%f" -o "shaders\%%~nf.spv"
    if errorlevel 1 (
        echo Failed to compile %%f
        pause
        exit /b 1
    )
)

echo Shaders compiled successfully!
echo.
echo Compiled shaders:
dir /b shaders\*.spv
echo.

echo Press any key to create setup.py and build the extension...
pause > nul

REM Create setup.py
echo Creating setup.py...
(
echo from setuptools import setup
echo from pybind11.setup_helpers import Pybind11Extension
echo import sys
echo import os
echo.
echo source_file = r'%CD:\=/%/src/vulkan_backend.cpp'
echo onnx_proto_cc = r'%ONNX_REPO_PATH:\=/%/onnx/onnx.pb.cc'
echo.
echo ext_modules = [
echo     Pybind11Extension(
echo         "vulkan_backend",
echo         [source_file, onnx_proto_cc],
echo         include_dirs=[
echo             r"%VULKAN_SDK:\=/%/Include",
echo             r"%ONNX_REPO_PATH:\=/%",
echo             r"%ONNX_REPO_PATH:\=/%/onnx",
echo             r"%PROTOC_DIR:\=/%/include",
echo             "src",
echo         ],
echo         library_dirs=[
echo             r"%VULKAN_SDK:\=/%/Lib"
echo         ],
echo         libraries=["vulkan-1"],
echo         extra_compile_args=['/std:c++17', '/DPROTOBUF_USE_DLLS'] if sys.platform == 'win32' else ['-std=c++17'],
echo         define_macros=[('PROTOBUF_USE_DLLS', 1)],
echo     ^)
echo ]
echo.
echo setup(
echo     name="vulkan_backend",
echo     ext_modules=ext_modules,
echo     zip_safe=False,
echo ^)
) > setup.py

echo Setup.py created successfully.
echo.
echo Building extension...
python setup.py build_ext --inplace
if errorlevel 1 (
    echo Failed to build extension!
    echo.
    echo Build error details above.
    echo Check if all dependencies are properly installed and paths are correct.
    pause
    exit /b 1
)

echo.
echo Build process completed successfully!
echo.
echo Cleaning up and deactivating virtual environment...
deactivate

echo.
echo Build process completed!
echo Original directory: %ORIGINAL_DIR%
echo Current directory: %CD%
echo.
echo Press any key to exit...
pause > nul

cd "%ORIGINAL_DIR%"
endlocal