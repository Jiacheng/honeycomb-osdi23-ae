#pragma once

#include <absl/status/status.h>
#include <hip/hip_runtime_api.h>

#include "aes_buffer.h"
#include "opencl/hsa/types.h"

namespace ocl::hip {

class DeviceContext;

class AESDeviceSrc;
class AESDevice {
  public:
    absl::Status Initialize(DeviceContext *parent);
    void Close();

    absl::Status Key(const uint32_t *ukey);
    absl::Status IV(const uint32_t *iv);
    absl::Status Encrypt(ocl::hsa::gpu_addr_t dst,
                         const AESDeviceSrc &device_src);

  protected:
    absl::Status LoadModule(hipModule_t *module, std::string module_path);
    absl::Status LoadBinary();

    absl::Status EncryptSome(ocl::hsa::gpu_addr_t out, ocl::hsa::gpu_addr_t in,
                             const unsigned long len, bool aligned);

    static const int kNumFunctions = 3;
    static const char *kFunctions[];
    hipModule_t module_;
    hipFunction_t func_[kNumFunctions];

    ocl::hsa::gpu_addr_t mem_;
    ocl::hsa::gpu_addr_t ks_;
    ocl::hsa::gpu_addr_t iv_;

    unsigned long iv_start_;

    DeviceContext *parent_;
};

} // namespace ocl::hip
