#include <dlfcn.h>
#include <random>

#include "memcpy.h"
#include "utils/monad_runner.h"
#include "opencl/hip/device_context.h"

namespace ocl::hip {

enum {
    kBufferSize = 1 << 24, // 16M
};

using namespace gpumpc;

absl::Status HipSecureMemcpy::Initialize(DeviceContext *parent) {
    parent_ = parent;

    // TODO: Diffe-Hellman
    const unsigned kMagic = 0xdeadbeaf;
    const unsigned ukey[kAES256KeySizeInWord] = {
        kMagic, kMagic, kMagic, kMagic, kMagic, kMagic, kMagic, kMagic};

    // Initialize host
    // ukey
    memcpy(ukey_, ukey, sizeof(ukey));

    // iv
    std::default_random_engine gen;
    std::uniform_int_distribution<unsigned> dist(
        std::numeric_limits<unsigned>::min(),
        std::numeric_limits<unsigned>::max());

    iv_[0] = dist(gen);
    iv_[1] = dist(gen);
    iv_[2] = iv_[3] = 0;

    // buffer
    buffer_.reserve(kBufferSize);

    // openssl context
    if (!(ctx_ = EVP_CIPHER_CTX_new())) {
        return absl::InvalidArgumentError("can not new openssl context");
    }
    if (1 != EVP_EncryptInit_ex(ctx_, EVP_aes_256_ctr(), NULL,
                                reinterpret_cast<const unsigned char *>(ukey_),
                                reinterpret_cast<const unsigned char *>(iv_))) {
        return absl::InvalidArgumentError("can not init openssl");
    }

    // Initialize device
    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner
        .Run([&]() {
            auto ret = parent_->GetMemoryManager()->hipMalloc(&hip_buffer_,
                                                              kBufferSize);
            if (ret != hipSuccess) {
                return absl::InvalidArgumentError("can not malloc");
            }
            return absl::OkStatus();
        })
        .Run([&]() { return aes_device_.Initialize(parent_); })
        .Run([&]() { return aes_device_.Key(ukey_); })
        .Run([&]() { return aes_device_.IV(iv_); });

    return runner.code();
}

void HipSecureMemcpy::Destroy() {
    EVP_CIPHER_CTX_free(ctx_);
    aes_device_.Close();
    auto _ = parent_->GetMemoryManager()->hipFree(hip_buffer_);
    (void)_;
}

absl::Status HipSecureMemcpy::OpenSSLUpdate(unsigned char *dst,
                                            const unsigned char *src,
                                            size_t len) {
    const unsigned char *aligned_src = src;
    size_t aligned_len = len;
    unsigned char *aligned_dst = dst;

    // pad unaligned input
    unsigned char src_buf[kAESBlockSize], dst_buf[kAESBlockSize];
    if (len < kAESBlockSize) {
        memcpy(src_buf, src, len);
        aligned_src = src_buf;
        aligned_dst = dst_buf;
        aligned_len = kAESBlockSize;
    }

    int updated_len;
    if (1 != EVP_EncryptUpdate(ctx_, aligned_dst, &updated_len, aligned_src,
                               aligned_len)) {
        return absl::InvalidArgumentError("can not encrypt");
    }

    if (len < kAESBlockSize) {
        memcpy(dst, dst_buf, len);
    }
    return absl::OkStatus();
}

absl::Status HipSecureMemcpy::HostEncrypt(const AESBuffer &dst,
                                          const AESBuffer &src) {
    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner
        .Run([&]() {
            if (src.IsHead()) {
                return OpenSSLUpdate(
                    reinterpret_cast<unsigned char *>(dst.GetHead()),
                    reinterpret_cast<unsigned char *>(src.GetHead()),
                    src.GetHeadLen());
            }
            return absl::OkStatus();
        })
        .Run([&]() {
            if (src.IsAligned()) {
                return OpenSSLUpdate(
                    reinterpret_cast<unsigned char *>(dst.GetAligned()),
                    reinterpret_cast<unsigned char *>(src.GetAligned()),
                    src.GetAlignedLen());
            }
            return absl::OkStatus();
        })
        .Run([&]() {
            if (src.IsTail()) {
                return OpenSSLUpdate(
                    reinterpret_cast<unsigned char *>(dst.GetTail()),
                    reinterpret_cast<unsigned char *>(src.GetTail()),
                    src.GetTailLen());
            }
            return absl::OkStatus();
        });
    return runner.code();
}

absl::Status HipSecureMemcpy::MemcpyHtoD(ocl::hsa::gpu_addr_t dst, void *src,
                                         size_t len) {
    MonadRunner<absl::Status> runner(absl::OkStatus());

    size_t sent = 0;
    while (sent != len) {
        size_t to_send = std::min(size_t(kBufferSize), len - sent);
        AESDeviceSrc device_src(reinterpret_cast<uintptr_t>(dst) + sent,
                                to_send);
        AESBuffer host_src(reinterpret_cast<uintptr_t>(src) + sent, device_src);
        AESBuffer host_dst(reinterpret_cast<uintptr_t>(buffer_.data()),
                           device_src);
        runner.Run([&]() { return HostEncrypt(host_dst, host_src); })
            .Run([&]() {
                auto hipError = parent_->GetMemoryManager()->hipMemcpyHtoD(
                    dst + sent, buffer_.data(), to_send);
                if (hipError != hipSuccess) {
                    return absl::InvalidArgumentError("can not memcpy");
                }
                return absl::OkStatus();
            })
            .Run([&]() {
                // inplace decrypt
                return aes_device_.Encrypt(
                    reinterpret_cast<uintptr_t>(dst) + sent, device_src);
            });
        sent += to_send;
    }
    return runner.code();
}

absl::Status HipSecureMemcpy::MemcpyDtoH(void *dst, ocl::hsa::gpu_addr_t src,
                                         size_t len) {
    MonadRunner<absl::Status> runner(absl::OkStatus());

    size_t sent = 0;
    while (sent != len) {
        size_t to_send = std::min(size_t(kBufferSize), len - sent);
        AESDeviceSrc device_src(reinterpret_cast<uintptr_t>(src) + sent,
                                to_send);
        AESBuffer device_dst(reinterpret_cast<uintptr_t>(hip_buffer_),
                             device_src);
        AESBuffer host(reinterpret_cast<uintptr_t>(dst) + sent, device_src);
        runner
            .Run([&]() { return aes_device_.Encrypt(hip_buffer_, device_src); })
            .Run([&]() {
                auto hipError = parent_->GetMemoryManager()->hipMemcpyDtoH(
                    reinterpret_cast<char *>(dst) + sent, hip_buffer_, to_send);
                if (hipError != hipSuccess) {
                    return absl::InvalidArgumentError("can not memcpy");
                }
                return absl::OkStatus();
            })
            .Run([&]() {
                // inplace decrypt
                return HostEncrypt(host, host);
            });
        sent += to_send;
    }
    return runner.code();
}

} // namespace ocl::hip
