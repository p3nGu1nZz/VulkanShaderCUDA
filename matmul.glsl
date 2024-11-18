#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer InputA { float a[]; };
layout(binding = 1) buffer InputB { float b[]; };
layout(binding = 2) buffer OutputC { float c[]; };

layout(push_constant) uniform MatMulParams {
    uint M; // Rows of A
    uint N; // Columns of B
    uint K; // Columns of A
} params;

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (row < params.M && col < params.N) {
        float value = 0.0;
        for (uint k = 0; k < params.K; ++k) {
            value += a[row * params.K + k] * b[k * params.N + col];
        }
        c[row * params.N + col] = value;
    }
}
