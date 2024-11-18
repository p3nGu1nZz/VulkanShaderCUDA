#!/bin/bash

# Compile GLSL shaders into SPIR-V
glslangValidator -V add.glsl -o add.spv
glslangValidator -V mul.glsl -o mul.spv
glslangValidator -V matmul.glsl -o matmul.spv
glslangValidator -V relu.glsl -o relu.spv
glslangValidator -V sigmoid.glsl -o sigmoid.spv
glslangValidator -V softmax.glsl -o softmax.spv
glslangValidator -V conv2d.glsl -o conv2d.spv
glslangValidator -V pooling.glsl -o pooling.spv

echo "Shaders compiled successfully!"
