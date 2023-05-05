#pragma once

#include "aes_buffer.h"
#include "aes_device.h"
#include "secure_memcpy.h"

#include <vector>

#include <absl/status/status.h>
#include <hip/hip_runtime.h>

#include <openssl/evp.h>

namespace ocl::hip {

class DeviceContext;

class HipSecureMemcpy {
  public:
    absl::Status Initialize(DeviceContext *parent);
    void Destroy();

    absl::Status MemcpyHtoD(ocl::hsa::gpu_addr_t dst, void *src,
                            std::size_t len);
    absl::Status MemcpyDtoH(void *dst, ocl::hsa::gpu_addr_t src,
                            std::size_t len);

  protected:
    absl::Status HostEncrypt(const AESBuffer &dst, const AESBuffer &src);
    absl::Status OpenSSLUpdate(unsigned char *dst, const unsigned char *src,
                               size_t len);

    DeviceContext *parent_;
    AESDevice aes_device_;

    ocl::hsa::gpu_addr_t hip_buffer_;

    EVP_CIPHER_CTX *ctx_;
    unsigned ukey_[kAES256KeySizeInWord];
    unsigned iv_[kAESIVSizeInWord];
    std::vector<unsigned char> buffer_;
};

} // namespace ocl::hip
