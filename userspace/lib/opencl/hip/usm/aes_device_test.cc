#include "aes_buffer.h"
#include "aes_device.h"
#include "secure_memcpy.h"

#include "opencl/hip/device_context.h"
#include "utils/monad_runner.h"

#include <functional>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <initializer_list>
#include <string_view>

#include <openssl/evp.h>

using namespace gpumpc;
using namespace ocl::hip;
using namespace ocl::hsa;

class AESDeviceTest : public ::testing::Test {
  protected:
    virtual void SetUp() override;
    virtual void TearDown() override;

    DeviceContext *ctx_;
    AESDevice aes_device;
    enum {
        kSize = (1 << 10),
        kMemSize = (3 << 10),
    };
    gpu_addr_t mem_;
};

void AESDeviceTest::SetUp() {
    ctx_ = GetCurrentDeviceContext();
    ASSERT_TRUE(ctx_);

    // aes_device = ctx_->GetSecureMemcpy()->GetAESDevice();
    auto stat = aes_device.Initialize(ctx_);
    ASSERT_TRUE(stat.ok());

    // 2K
    // 1K for plain text and cipher text
    // 1K for decrypted plain text
    ASSERT_EQ(ctx_->GetMemoryManager()->hipMalloc(&mem_, kMemSize), hipSuccess);
}

void AESDeviceTest::TearDown() {
    if (mem_) {
        ASSERT_EQ(ctx_->GetMemoryManager()->hipFree(mem_), hipSuccess);
    }

    aes_device.Close();
}

TEST_F(AESDeviceTest, TestAES) {
    enum {
        kBlocks = 10,
        kMagic = 0xdeadbeaf,
    };

    ASSERT_LE(kAESBlockSize * kBlocks, kSize);
    ASSERT_LE(3 * kSize, kMemSize);

    uint32_t ukey[kAES256KeySizeInWord] = {kMagic, kMagic, kMagic, kMagic};
    uint32_t iv[kAESIVSizeInWord] = {kMagic, kMagic, 0, 0};

    uint32_t plain[kAESBlockSizeInWord * kBlocks];

    for (int i = 0; i != kAESBlockSizeInWord * kBlocks; ++i) {
        plain[i] = kMagic;
    }

    gpu_addr_t plain_addr = mem_;
    gpu_addr_t decrypted_addr = plain_addr + kSize;

    AESDeviceSrc device_src(plain_addr, kAESBlockSize * kBlocks);

    ASSERT_EQ(ctx_->GetMemoryManager()->hipMemcpyHtoD(plain_addr, plain,
                                                      kAESBlockSize * kBlocks),
              hipSuccess);

    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return aes_device.Key(ukey); })
        .Run([&]() { return aes_device.IV(iv); })
        .Run([&]() {
            // in-place
            return aes_device.Encrypt(plain_addr, device_src);
        })
        .Run([&]() {
            // reset IV to decrypt in device
            return aes_device.IV(iv);
        })
        .Run([&]() { return aes_device.Encrypt(decrypted_addr, device_src); });
    ASSERT_TRUE(runner.code().ok());

    uint32_t decrypted[kAESBlockSizeInWord * kBlocks];
    auto stat = ctx_->GetMemoryManager()->hipMemcpyDtoH(
        decrypted, decrypted_addr, kAESBlockSize * kBlocks);
    ASSERT_EQ(stat, hipSuccess);

    for (int i = 0; i != kAESBlockSizeInWord * kBlocks; ++i) {
        ASSERT_EQ(plain[i], decrypted[i]);
    }
}

TEST_F(AESDeviceTest, TestValidateAgainstAESGCM) {
    enum {
        kBlocks = 10,
        kMagic = 0xdeadbeaf,
    };

    ASSERT_LE(kAESBlockSize * kBlocks, kSize);
    ASSERT_LE(3 * kSize, kMemSize);

    uint32_t ukey[kAES256KeySizeInWord] = {kMagic, kMagic, kMagic, kMagic};

    // iv for CTR mode
    uint32_t iv[kAESIVSizeInWord] = {kMagic, kMagic, 0, 0};
    // NOTE: to validate CTR against GCM, iv should be initialized
    // with some advancement, because GCM consumes two iv before
    // actually encrypting on the data
    uint8_t *iv_u8 = reinterpret_cast<uint8_t *>(iv);
    iv_u8[15] = 2; // assumed iv_u8[15] == 0

    // iv for GCM mode
    uint32_t iv_gcm[kAESIVSizeInWord] = {kMagic, kMagic, 0, 0};
    uint8_t *iv_gcm_u8 = reinterpret_cast<uint8_t *>(iv_gcm);

    uint32_t plain[kAESBlockSizeInWord * kBlocks];

    for (int i = 0; i != kAESBlockSizeInWord * kBlocks; ++i) {
        plain[i] = kMagic;
    }

    gpu_addr_t plain_addr = mem_;

    AESDeviceSrc device_src(plain_addr, kAESBlockSize * kBlocks);

    ASSERT_EQ(ctx_->GetMemoryManager()->hipMemcpyHtoD(plain_addr, plain,
                                                      kAESBlockSize * kBlocks),
              hipSuccess);

    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return aes_device.Key(ukey); })
        .Run([&]() { return aes_device.IV(iv); })
        .Run([&]() {
            // in-place
            return aes_device.Encrypt(plain_addr, device_src);
        });
    ASSERT_TRUE(runner.code().ok());

    uint32_t encrypted[kAESBlockSizeInWord * kBlocks];
    auto stat = ctx_->GetMemoryManager()->hipMemcpyDtoH(
        encrypted, plain_addr, kAESBlockSize * kBlocks);
    ASSERT_EQ(stat, hipSuccess);

    // GCM encrypt
    EVP_CIPHER_CTX *ctx;
    uint32_t encrypted_gcm[kAESBlockSizeInWord * kBlocks];
    int encrypted_gcm_len;

    if (!(ctx = EVP_CIPHER_CTX_new())) {
        FAIL();
    }
    if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL,
                                reinterpret_cast<unsigned char *>(ukey),
                                iv_gcm_u8)) {
        FAIL();
    }

    if (1 != EVP_EncryptUpdate(
                 ctx, reinterpret_cast<unsigned char *>(encrypted_gcm),
                 &encrypted_gcm_len, reinterpret_cast<unsigned char *>(plain),
                 kAESBlockSize * kBlocks)) {
        FAIL();
    }

    ASSERT_EQ(encrypted_gcm_len, kAESBlockSize * kBlocks);

    for (int i = 0; i != kAESBlockSizeInWord * kBlocks; ++i) {
        ASSERT_EQ(encrypted_gcm[i], encrypted[i]);
    }
}
