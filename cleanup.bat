@echo off
setlocal EnableDelayedExpansion

echo This script will clean all build artifacts and test a fresh setup.
echo.
echo WARNING: This will delete:
echo - build directory
echo - protoc directory
echo - ONNX directory
echo - bin directory
echo - downloaded files
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

REM Store the original directory
set "PROJECT_ROOT=%CD%"

REM Clean directories and files
echo Cleaning previous build artifacts...
if exist "build" rd /s /q "build"
if exist "protoc-28.3" rd /s /q "protoc-28.3"
if exist "onnx" rd /s /q "onnx"
if exist "protobuf-28.3" rd /s /q "protobuf-28.3"
if exist "bin" rd /s /q "bin"
if exist "shaders\*.spv" del /f /q "shaders\*.spv"
if exist "*.zip" del /f /q "*.zip"