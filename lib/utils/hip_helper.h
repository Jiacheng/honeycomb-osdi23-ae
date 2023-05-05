#pragma once

#include <absl/status/status.h>
#include <hip/hip_runtime_api.h>

namespace gpumpc::experiment {

struct Dimension {
    unsigned x_;
    unsigned y_;
    unsigned z_;
    Dimension(unsigned x) {
        x_ = x;
        y_ = 1;
        z_ = 1;
    }
    explicit Dimension(unsigned x, unsigned y) {
        x_ = x;
        y_ = y;
        z_ = 1;
    }
    explicit Dimension(unsigned x, unsigned y, unsigned z) {
        x_ = x;
        y_ = y;
        z_ = z;
    }
};

template <class T>
static inline absl::Status LaunchKernel(hipFunction_t f, const Dimension &grid,
                                        const Dimension &block, const T &args) {
    size_t size = sizeof(T);
    const void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                           HIP_LAUNCH_PARAM_END};
    auto err = hipModuleLaunchKernel(f, grid.x_, grid.y_, grid.z_, block.x_,
                                     block.y_, block.z_, 0, nullptr, nullptr,
                                     const_cast<void **>(extra));
    if (err != hipSuccess) {
        return absl::InvalidArgumentError("Failed to launch kernel");
    }
    return absl::OkStatus();
}

static inline absl::Status HipErrToStatus(hipError_t err) {
    if (err != hipSuccess) {
        return absl::InvalidArgumentError("HIP error");
    }
    return absl::OkStatus();
}

#define checkHipErrors(call)                                                   \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            printf("HIP error %d in %s:%d\n", err, __FUNCTION__, __LINE__);    \
            abort();                                                           \
        }                                                                      \
    } while (0)
} // namespace gpumpc::experiment
