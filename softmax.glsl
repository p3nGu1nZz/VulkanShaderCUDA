#version 450

layout(local_size_x = 256) in;

layout(binding = 0) buffer Input { float input[]; };
layout(binding = 1) buffer Output { float output[]; };

layout(push_constant) uniform SoftmaxParams {
    uint length;
} params;

shared float sharedSum;

void main() {
    uint idx = gl_GlobalInvocationID.x;

    // Exponentiation
    float expValue = exp(input[idx]);

    // Compute shared sum
    atomicAdd(sharedSum, expValue);
    barrier();

    // Normalize
    output[idx] = expValue / sharedSum;
}
