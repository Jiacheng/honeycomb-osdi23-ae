#pragma once

#define GLOBAL_SIZE_0 1024
#define GLOBAL_SIZE_1 1
#define GLOBAL_SIZE_2 1
#define GLOBAL_ID_0 (threadIdx.x + blockIdx.x * 256)
#define GLOBAL_ID_1 0
#define GLOBAL_ID_2 0

#define DEFINE_CONV2D_ARGS_WITH_DILATIONS(_BATCH, _IN_CHANNEL, \
    _OUT_CHANNEL, _IN_H, _IN_W, _KERNEL_H, _KERNEL_W, _STRIDE_H, \
    _STRIDE_W,  _PADDING_H, _PADDING_W, _DILATION_H, _DILATION_W) \
        const int BATCH = (_BATCH); \
        const int IN_CHANNEL = (_IN_CHANNEL); \
        const int OUT_CHANNEL = (_OUT_CHANNEL); \
        const int IN_H = (_IN_H); \
        const int IN_W = (_IN_W); \
        const int KERNEL_H = (_KERNEL_H); \
        const int KERNEL_W = (_KERNEL_W); \
        const int STRIDE_H = (_STRIDE_H); \
        const int STRIDE_W = (_STRIDE_W); \
        const int PADDING_H = (_PADDING_H); \
        const int PADDING_W = (_PADDING_W); \
        const int DILATION_W = (_DILATION_W); \
        const int DILATION_H = (_DILATION_H); \
        const int OUT_H = (IN_H - ((KERNEL_H - 1) * DILATION_H + 1) + PADDING_H * 2) / STRIDE_H + 1; \
        const int OUT_W = (IN_W - ((KERNEL_W - 1) * DILATION_W + 1) + PADDING_W * 2) / STRIDE_W + 1;

#define DEFINE_CONV2D_ARGS(_BATCH, _IN_CHANNEL, \
    _OUT_CHANNEL, _IN_H, _IN_W, _KERNEL_H, _KERNEL_W, _STRIDE_H, \
    _STRIDE_W,  _PADDING_H, _PADDING_W) \
        DEFINE_CONV2D_ARGS_WITH_DILATIONS((_BATCH), (_IN_CHANNEL), \
            (_OUT_CHANNEL), (_IN_H), (_IN_W), (_KERNEL_H), (_KERNEL_W), \
            (_STRIDE_H), (_STRIDE_W), (_PADDING_H), (_PADDING_W), 1, 1)

#define CONV2D_LOOPS do { \
    for (int nn = GLOBAL_ID_0; nn < BATCH; nn += GLOBAL_SIZE_0) { \
        for (int ff = GLOBAL_ID_1; ff < OUT_CHANNEL; ff += GLOBAL_SIZE_1) { \
            for (int yy = GLOBAL_ID_2; yy < OUT_H; yy += GLOBAL_SIZE_2) { \
                for (int xx = 0; xx < OUT_W; ++xx) { \
                    out[((((nn * OUT_CHANNEL) + ff) * OUT_H) + yy) * OUT_W + xx] = 0; \
                } \
            } \
        } \
    } \
\
    for (int rc = 0; rc < IN_CHANNEL; ++rc) { \
        for (int ry = 0; ry < KERNEL_H; ++ry) { \
            for (int rx = 0; rx < KERNEL_W; ++rx) { \
                for (int nn = GLOBAL_ID_0; nn < BATCH; nn += GLOBAL_SIZE_0) { \
                    for (int ff = GLOBAL_ID_1; ff < OUT_CHANNEL; ff += GLOBAL_SIZE_1) { \
                        for (int yy = GLOBAL_ID_2; yy < OUT_H; yy += GLOBAL_SIZE_2) { \
                            for (int xx = 0; xx < OUT_W; ++xx) { \
                                const int jj = yy * STRIDE_H + ry * DILATION_H; \
                                const int j = jj - PADDING_H; \
                                if (j >= 0 && j < IN_H) { \
                                    const int ii = xx * STRIDE_W + rx * DILATION_W; \
                                    const int i = ii - PADDING_W; \
                                    if (i >= 0 && i < IN_W) { \
                                        out[((((nn * OUT_CHANNEL) + ff) * OUT_H) + yy) * OUT_W + xx] += \
                                                in[((((nn * IN_CHANNEL) + rc) * IN_H) + j) * IN_W + i] * \
                                                ker[((((ff * IN_CHANNEL) + rc) * KERNEL_H) + ry) * KERNEL_W + rx]; \
                                    } \
                                } \
                            } \
                        } \
                    } \
                } \
            } \
        } \
    } \
} while(0)

extern "C" __global__ void convolution_forward_replacement1(
        const float *__restrict__ ker,
        const float *__restrict__ in,
        float *__restrict__ out,
        const void *__restrict__ p_desc_tuple) {

    DEFINE_CONV2D_ARGS(128, 64, 128, 3, 3, 3, 3, 2, 2, 1, 1);
    CONV2D_LOOPS;
}

extern "C" __global__ void convolution_forward_replacement2(
        const float *__restrict__ ker,
        const float *__restrict__ in,
        float *__restrict__ out,
        const void *__restrict__ p_desc_tuple) {

    DEFINE_CONV2D_ARGS(128, 64, 128, 3, 3, 1, 1, 2, 2, 0, 0);
    CONV2D_LOOPS;
}
