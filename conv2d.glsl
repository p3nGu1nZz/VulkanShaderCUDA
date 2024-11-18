#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer Input { float input[]; };
layout(binding = 1) buffer Weights { float weights[]; };
layout(binding = 2) buffer Output { float output[]; };

layout(push_constant) uniform Conv2DParams {
    uint inputWidth;
    uint inputHeight;
    uint filterWidth;
    uint filterHeight;
    uint stride;
} params;

void main() {
    uint x = gl_GlobalInvocationID.x * params.stride;
    uint y = gl_GlobalInvocationID.y * params.stride;
    uint filterX, filterY;

    float sum = 0.0;

    for (filterY = 0; filterY < params.filterHeight; ++filterY) {
        for (filterX = 0; filterX < params.filterWidth; ++filterX) {
            uint inputX = x + filterX;
            uint inputY = y + filterY;

            uint inputIdx = inputY * params.inputWidth + inputX;
            uint weightIdx = filterY * params.filterWidth + filterX;

            sum += input[inputIdx] * weights[weightIdx];
        }
    }

    uint outputIdx = gl_GlobalInvocationID.y * params.inputWidth + gl_GlobalInvocationID.x;
    output[outputIdx] = sum;
}
