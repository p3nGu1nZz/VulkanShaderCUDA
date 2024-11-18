#version 450

layout(local_size_x = 256) in;

layout(binding = 0) buffer Input { float input[]; };
layout(binding = 1) buffer Output { float output[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}
