#version 450

layout(local_size_x = 256) in;

layout(binding = 0) buffer InputA { float a[]; };
layout(binding = 1) buffer InputB { float b[]; };
layout(binding = 2) buffer OutputC { float c[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    c[idx] = a[idx] + b[idx];
}
