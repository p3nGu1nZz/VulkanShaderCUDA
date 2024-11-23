#ifndef PUSH_CONSTANTS_H
#define PUSH_CONSTANTS_H

#include <cstdint>

struct MatMulPushConstants {
    uint32_t M;
    uint32_t K;
    uint32_t N;
};

struct Conv2DPushConstants {
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t kernel_size;
    uint32_t padding;
    uint32_t stride;
};

struct SoftmaxPushConstants {
    uint32_t size;
};

struct MaxPoolPushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t batch_size;
    uint32_t poolSizeX;
    uint32_t poolSizeY;
    uint32_t strideX;
    uint32_t strideY;
};

struct BatchNormPushConstants {
    uint32_t size;
    float epsilon;
};

struct AddPushConstants {
    uint32_t total_elements;
};

struct ReLUPushConstants {
    uint32_t size;
};

struct SigmoidPushConstants {
    uint32_t size;
};

#endif // PUSH_CONSTANTS_H