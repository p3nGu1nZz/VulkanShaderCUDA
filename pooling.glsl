#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer Input { float input[]; };
layout(binding = 1) buffer Output { float output[]; };

layout(push_constant) uniform PoolingParams {
    uint inputWidth;
    uint inputHeight;
    uint poolWidth;
    uint poolHeight;
    uint stride;
    uint isMaxPool; // 1 for Max Pooling, 0 for Avg Pooling
} params;

void main() {
    uint x = gl_GlobalInvocationID.x * params.stride;
    uint y = gl_GlobalInvocationID.y * params.stride;

    float poolValue = params.isMaxPool == 1 ? -1e9 : 0.0;
    uint count = 0;

    for (uint i = 0; i < params.poolHeight; ++i) {
        for (uint j = 0; j < params.poolWidth; ++j) {
            uint inputX = x + j;
            uint inputY = y + i;

            uint inputIdx = inputY * params.inputWidth + inputX;

            if (params.isMaxPool == 1) {
                poolValue = max(poolValue, input[inputIdx]);
            } else {
                poolValue += input[inputIdx];
                count++;
            }
        }
    }

    uint outputIdx = gl_GlobalInvocationID.y * (params.inputWidth / params.stride) + gl_GlobalInvocationID.x;
    output[outputIdx] = params.isMaxPool == 1 ? poolValue : poolValue / count;
}
