#include <hip/hip_runtime.h>

__global__ extern "C" void addalpha(float *__restrict__ c,
                                    const float *__restrict__ a,
                                    const float *__restrict__ b, float alpha) {
    enum {
        kTotalElement = 65536,
        kBlockSize = 256,
        kGridSize = kTotalElement / kBlockSize,
    };

#pragma unroll
    for (int i = 0; i < kGridSize / kBlockSize; i++) {
        int idx =
            threadIdx.x + blockIdx.x * kBlockSize + i * kGridSize * kBlockSize;
        c[idx] = a[idx] + b[idx] * alpha;
    }
}

template <class Functor>
__device__ static inline void
ElementWiseOp(int N, float arg, float *__restrict__ c,
              const float *__restrict__ a, const float *__restrict__ b) {
    enum {
        kBlockSize = 256,
        kParallelism = 4,
    };

    int idx = threadIdx.x + blockIdx.x * kBlockSize * kParallelism;

    float lhs[kParallelism];
    float rhs[kParallelism];

#pragma unroll
    for (int i = 0; i < kParallelism; i++) {
        if (idx + i * kBlockSize < N) {
            lhs[i] = a[idx + i * kBlockSize];
            rhs[i] = b[idx + i * kBlockSize];
        }
    }

#pragma unroll
    for (int i = 0; i < kParallelism; i++) {
        if (idx + i * kBlockSize < N) {
            c[idx + i * kBlockSize] = Functor::Run(arg, lhs[i], rhs[i]);
        }
    }
}

struct AddAlphaFunctor {
    __device__ static inline float Run(float alpha, float lhs, float rhs) {
        return lhs + rhs * alpha;
    }
};

extern "C" __launch_bounds__(256) __global__
    void addalpha_generic(int N, float alpha, float *__restrict__ c,
                          const float *__restrict__ a,
                          const float *__restrict__ b) {
    ElementWiseOp<AddAlphaFunctor>(N, alpha, c, a, b);
}

extern "C" __global__ void relu(float *__restrict__ res,
                                const float *__restrict__ src) {
    enum {
        kTotalElement = 65536,
        kGridSize = 64,
        kBlockSize = 256,
    };

#pragma unroll
    for (int i = 0; i < kTotalElement / kGridSize / kBlockSize; i++) {
        int idx =
            threadIdx.x + blockIdx.x * kBlockSize + i * kGridSize * kBlockSize;
        res[idx] = (src[idx] > 0.0f) ? src[idx] : 0.0f;
    }
}

struct ReluFunctor {
    __device__ static inline float Run(float _, float lhs, float rhs) {
        return lhs > 0.0f ? lhs : 0.0f;
    }
};
extern "C" __global__ void relu_generic(int N, float *res, const float *src) {
    ElementWiseOp<ReluFunctor>(N, 0, res, src, src);
}

extern "C" __global__ void
batch_norm_opt_resnet1(const float4 *__restrict__ in, float4 *__restrict__ out,
                       const float *__restrict__ est_mean,
                       const float *__restrict__ est_variance,
                       const float *__restrict__ scale,
                       const float *__restrict__ bias, double epsilon) {
    enum {
        kBlockSizeX = 128,
    };
    typedef float FloatPrec;
    typedef float Float;

    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * kBlockSizeX;

    FloatPrec mean = est_mean[tid];
    FloatPrec variance = est_variance[tid];
    FloatPrec pscale = scale[tid];
    FloatPrec pbias = bias[tid];
    FloatPrec inv_variance = rsqrt(fabs(variance + epsilon));

    float4 in_val = in[gid];
    float4 inhat;
    const float *v = reinterpret_cast<const float *>(&in_val);
    float *r = reinterpret_cast<float *>(&inhat);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float t = (v[i] - mean) * inv_variance;
        r[i] = pscale * t + pbias;
    }
    out[gid] = inhat;
}

template <typename scalar_t, typename accscalar_t>
__device__ static inline void max_pool_forward_nchw_kernel3_stride2_pad1_dilation1(
    const int nthreads, const scalar_t *bottom_data, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, scalar_t *top_data, int64_t *top_mask) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
         index += blockDim.x * gridDim.x) {
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;
        int hstart = ph * 2 - 1;
        int wstart = pw * 2 - 1;
        int hend = min(hstart + 3, height);
        int wend = min(wstart + 3, width);
        hstart = max (hstart, 0);
        wstart = max (wstart, 0);
        accscalar_t maxval =
            std::numeric_limits<accscalar_t>::lowest(); // -Infinity
        int maxidx = hstart * width + wstart;
        const scalar_t *btm_data =
            bottom_data + (n * channels + c) * height * width;
        for (int h = hstart; h < hend; h += 1) {
            for (int w = wstart; w < wend; w += 1) {
                scalar_t val = btm_data[h * width + w];
                if ((static_cast<accscalar_t>(val) > maxval) ||
                    std::isnan(val)) {
                    maxidx = h * width + w;
                    maxval = static_cast<accscalar_t>(val);
                }
            }
        }
        top_data[index] = static_cast<accscalar_t>(maxval);
        top_mask[index] = maxidx;
    }
}

// CUDA: grid stride looping
//
// int64_t _i_n_d_e_x specifically prevents overflow in the loop increment.
// If input.numel() < INT_MAX, _i_n_d_e_x < INT_MAX, except after the final
// iteration of the loop where _i_n_d_e_x += blockDim.x * gridDim.x can be
// greater than INT_MAX.  But in that case _i_n_d_e_x >= n, so there are no
// further iterations and the overflowed value in i=_i_n_d_e_x is not used.
#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)

template <typename scalar_t, typename accscalar_t>
__device__ static inline void avg_pool2d_out_cuda_frame(const int nthreads,
    const scalar_t* const bottom_data, const int64_t channels,
    const int64_t height, const int64_t width, const int64_t pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const top_data, const int divisor_override,
    const bool count_include_pad, const bool use_divisor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);

    if (hstart >= hend || wstart >= wend) {
      top_data[index] = scalar_t(0);
      continue;
    }

    accscalar_t aveval = accscalar_t(0);
    const scalar_t* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if(count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
  }
}
// Dummy variables due to internal ABI changes from PyTorch
// e33cd8f3826087074e22382dec10e65c67e25b8c
extern "C" __global__ void max_pool_forward_nchw(
    const int nthreads, const float *bottom_data, const int _,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    float *top_data, int64_t *top_mask) {
    max_pool_forward_nchw_kernel3_stride2_pad1_dilation1<float, float>(
        nthreads, bottom_data, channels, height, width, pooled_height,
        pooled_width, top_data, top_mask);
}


__global__ extern "C"  void avg_pool_forward_nchw(
    float *input, float *output
) {
    avg_pool2d_out_cuda_frame<float, float>(
        512,
        input, 512, 7, 7, 1, 1,
        7, 7,
        7, 7,
        0, 0,
        output, 0,
        false, false
    );
    /*adaptive_average_pool<float>(
        input, output,
        isizeH, isizeW,
        osizeH, osizeW,
        256 * isizeH * isizeW, isizeW, 0
    );*/
}

__global__ extern "C" void elementwise_add(int N, float alpha,
                                           float *__restrict__ c,
                                           const float *__restrict__ a,
                                           const float *__restrict__ b) {}

#include "conv_replacement.cuh"