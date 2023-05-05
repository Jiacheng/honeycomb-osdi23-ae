#include "device_context.h"
#include "opencl/hsa/assert.h"
#include "opencl/hsa/runtime_options.h"
#include "opencl/hsa/types.h"
#include <hip/hip_runtime_api.h>

using namespace ocl;
using namespace ocl::hip;

#define EXPORT_API __attribute__((visibility("default")))
#define ENSURE_CTX                                                             \
    auto ctx = GetCurrentDeviceContext();                                      \
    if (!ctx) {                                                                \
        return hipErrorInvalidContext;                                         \
    }

static inline bool IsSecuryMemoryEnabled() {
    auto opt = GetRuntimeOptions();
    return opt && opt->IsSecureMemcpy();
}

using ocl::hsa::gpu_addr_t;

extern "C" {

EXPORT_API hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void *src,
                                    size_t sizeBytes) {
    ENSURE_CTX;
    if (IsSecuryMemoryEnabled()) {
        auto usm = ctx->GetSecureMemcpy();
        auto stat =
            usm->MemcpyHtoD(reinterpret_cast<gpu_addr_t>(dst), src, sizeBytes);
        if (stat.ok()) {
            return hipSuccess;
        } else {
            return hipErrorInvalidValue;
        }
    } else {
        auto mm = ctx->GetMemoryManager();
        return mm->hipMemcpyHtoD(reinterpret_cast<gpu_addr_t>(dst), src,
                                 sizeBytes);
    }
}

EXPORT_API hipError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src,
                                    size_t sizeBytes) {
    ENSURE_CTX;
    if (IsSecuryMemoryEnabled()) {
        auto usm = ctx->GetSecureMemcpy();
        auto stat =
            usm->MemcpyDtoH(dst, reinterpret_cast<gpu_addr_t>(src), sizeBytes);
        if (stat.ok()) {
            return hipSuccess;
        } else {
            return hipErrorInvalidValue;
        }
    } else {
        auto mm = ctx->GetMemoryManager();
        return mm->hipMemcpyDtoH(dst, reinterpret_cast<gpu_addr_t>(src),
                                 sizeBytes);
    }
}

EXPORT_API hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src,
                                    size_t sizeBytes) {
    ENSURE_CTX;
    auto mm = ctx->GetMemoryManager();
    return mm->hipMemcpyDtoD(reinterpret_cast<gpu_addr_t>(dst),
                             reinterpret_cast<gpu_addr_t>(src), sizeBytes);
}

EXPORT_API hipError_t hipFree(void *ptr) {
    ENSURE_CTX;
    auto mm = ctx->GetMemoryManager();
    return mm->hipFree(reinterpret_cast<gpu_addr_t>(ptr));
}

EXPORT_API hipError_t hipMalloc(void **ptr, size_t size) {
    ENSURE_CTX;
    auto mm = ctx->GetMemoryManager();
    return mm->hipMalloc(reinterpret_cast<gpu_addr_t *>(ptr), size);
}

EXPORT_API hipError_t hipModuleLoadData(hipModule_t *module,
                                        const void *image) {
    ENSURE_CTX;
    auto compute = ctx->GetComputeContext();
    return compute->hipModuleLoadData(module,
                                      reinterpret_cast<const char *>(image));
}

EXPORT_API hipError_t hipModuleUnload(hipModule_t module) {
    ENSURE_CTX;
    auto compute = ctx->GetComputeContext();
    return compute->hipModuleUnload(module);
}

EXPORT_API hipError_t hipModuleGetFunction(hipFunction_t *function,
                                           hipModule_t module,
                                           const char *kname) {
    ENSURE_CTX;
    auto compute = ctx->GetComputeContext();
    return compute->hipModuleGetFunction(function, module, kname);
}

EXPORT_API hipError_t hipModuleLaunchKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void **kernelParams, void **extra) {
    ENSURE_CTX;
    auto compute = ctx->GetComputeContext();
    if (extra[0] != HIP_LAUNCH_PARAM_BUFFER_POINTER ||
        extra[2] != HIP_LAUNCH_PARAM_BUFFER_SIZE ||
        extra[4] != HIP_LAUNCH_PARAM_END) {
        return hipErrorInvalidValue;
    }
    const void *args = extra[1];
    size_t size = *reinterpret_cast<const size_t *>(extra[3]);
    return compute->hipModuleLaunchKernel(
        f, gridDimX * blockDimX, gridDimY * blockDimY, gridDimZ * blockDimZ,
        blockDimX, blockDimY, blockDimZ, args, size);
}

EXPORT_API hipError_t hipDeviceSynchronize(void) {
    ENSURE_CTX;
    auto compute = ctx->GetComputeContext();
    return compute->ReleaseDeviceMemoryFence();
}
}
