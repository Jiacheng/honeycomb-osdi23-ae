#pragma once

#include "opencl/hsa/platform.h"
#include "opencl/hsa/signals.h"
#include "opencl/hsa/types.h"
#include "usm/memcpy.h"
#include <absl/status/status.h>
#include <hip/hip_runtime_api.h>
#include <memory>

namespace ocl::hsa {
class SignalPool;
}

namespace ocl::hip {

class DeviceContext;

class HipMemoryManagerBase {
  public:
    virtual hipError_t hipMalloc(ocl::hsa::gpu_addr_t *ptr, size_t size) = 0;
    virtual hipError_t hipFree(ocl::hsa::gpu_addr_t ptr) = 0;
    virtual hipError_t hipMemcpyDtoH(void *dst, ocl::hsa::gpu_addr_t src,
                                     size_t size) = 0;
    virtual hipError_t hipMemcpyHtoD(ocl::hsa::gpu_addr_t dst, const void *src,
                                     size_t size) = 0;
    virtual hipError_t hipMemcpyDtoD(ocl::hsa::gpu_addr_t dst,
                                     ocl::hsa::gpu_addr_t src, size_t size) = 0;
    virtual absl::Status Initialize() = 0;
    virtual absl::Status Destroy() = 0;
    virtual ~HipMemoryManagerBase() = default;

    // Memcpy with the custom HSA implementation
    static std::unique_ptr<HipMemoryManagerBase>
    NewHsaMemoryManager(DeviceContext *parent, hsa::Signal *barrier);
};

class HipComputeContextBase {
  public:
    virtual hipError_t hipModuleLoadData(hipModule_t *module,
                                         const char *data) = 0;
    virtual hipError_t hipModuleGetFunction(hipFunction_t *function,
                                            hipModule_t module,
                                            const char *kname) = 0;
    virtual hipError_t hipModuleLaunchKernel(
        hipFunction_t f, unsigned globalDimX, unsigned globalDimY,
        unsigned globalDimZ, unsigned blockDimX, unsigned blockDimY,
        unsigned blockDimZ, const void *args, size_t arg_size) = 0;
    virtual hipError_t hipModuleUnload(hipModule_t module) = 0;
    virtual absl::Status Initialize() = 0;
    virtual absl::Status Destroy() = 0;
    virtual ~HipComputeContextBase() = default;

    virtual void AddSystemScopeAcquire() { system_scope_acquire_ = true; }
    virtual hipError_t ReleaseDeviceMemoryFence() = 0;

    // Memcpy with the custom HSA implementation
    static std::unique_ptr<HipComputeContextBase>
    NewHsaComputeContext(DeviceContext *parent, hsa::Signal *barrier);

  protected:
    bool pending_dispatch_;
    bool system_scope_acquire_;
};

//
// Get the current device context. The HIP API allows switching devices,
// although the current implementation only supports a single device.
DeviceContext *GetCurrentDeviceContext();

class DeviceContext {
  public:
    HipMemoryManagerBase *GetMemoryManager() const { return mm_.get(); }
    HipComputeContextBase *GetComputeContext() const { return compute_.get(); }
    HipSecureMemcpy *GetSecureMemcpy() const { return usm_.get(); }
    friend DeviceContext *GetCurrentDeviceContext();
    ~DeviceContext();
    bool IsInitialized() const { return initialized_; }
    hsa::Device *GetImplementation() const { return impl_; }
    hsa::SignalPool *GetSignalPool() const { return signal_pool_.get(); }

  private:
    DeviceContext(const DeviceContext &) = delete;
    DeviceContext &operator=(const DeviceContext &) = delete;
    DeviceContext(DeviceContext &&) = delete;

    DeviceContext();
    absl::Status Initialize();

    bool initialized_;

    hsa::Device *impl_;
    std::unique_ptr<HipMemoryManagerBase> mm_;
    std::unique_ptr<HipComputeContextBase> compute_;
    std::unique_ptr<hsa::SignalPool> signal_pool_;
    std::unique_ptr<HipSecureMemcpy> usm_;
    hsa::Signal *barrier_;
};

} // namespace ocl::hip
